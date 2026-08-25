"""Cost accounting, and the budget that makes a runaway loop stop itself.

This module is small and load-bearing. An agent loop that retries on failure
will happily spend a month's budget in an afternoon, and the failure mode is
not a crash -- it is a correct-looking run and an invoice. A budget that is
merely *reported* does not prevent that. A budget that *raises* does.

Three decisions worth stating:

**Prices are per million tokens, supplied by the caller.** Hardcoding a price
table means the library ships stale numbers the day a provider reprices, and a
stale price silently understates spend. The caller owns the table because the
caller knows which contract they are on.

**Cached input tokens are priced separately.** Prompt-cache reads are commonly
an order of magnitude cheaper than fresh input. Folding them into the input
rate overstates cost enough to hide whether caching is paying for itself,
which is the only reason to measure it.

**The check is exact, not floating-point-tolerant.** Money is counted in
`Decimal` internally: 0.1 + 0.2 != 0.3 in binary floating point, and a ledger
that drifts by fractions of a cent per call is a ledger nobody trusts after a
million calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal

TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)


class BudgetExceeded(RuntimeError):
    """Raised when recorded spend has passed the ledger's budget.

    Deliberately a hard error rather than a warning. The whole value of a
    budget is that it terminates the loop; a warning in a log nobody reads
    during an unattended run is indistinguishable from no budget at all.
    """

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"budget exceeded: spent ${spent_usd:.6f} of ${budget_usd:.6f}"
        )


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens for one model.

    `cached_input_per_mtok` defaults to the full input rate rather than to
    zero: assuming a discount the caller did not confirm understates spend,
    and understating spend is the one direction a cost tool must never err in.
    """

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        if cached_tokens > prompt_tokens:
            raise ValueError(
                f"cached_tokens ({cached_tokens}) exceeds prompt_tokens "
                f"({prompt_tokens})"
            )
        cached_rate = (
            self.input_per_mtok
            if self.cached_input_per_mtok is None
            else self.cached_input_per_mtok
        )
        fresh = prompt_tokens - cached_tokens
        total = (
            Decimal(str(self.input_per_mtok)) * Decimal(fresh)
            + Decimal(str(cached_rate)) * Decimal(cached_tokens)
            + Decimal(str(self.output_per_mtok)) * Decimal(completion_tokens)
        )
        return total / TOKENS_PER_PRICE_UNIT


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded call. Kept whole so spend can be attributed after the fact.

    Storing only a running total is the cheaper design and the wrong one: when
    a bill surprises you, the question is always *which model, which call*, and
    a scalar cannot answer it.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: Decimal


@dataclass
class CostLedger:
    """Running spend across calls, with an optional hard budget.

    Usage::

        ledger = CostLedger({"gpt-4o-mini": ModelPrice(0.15, 0.60)},
                            budget_usd=1.00)
        ledger.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        ledger.check_budget()   # raises BudgetExceeded once over

    `record()` does not raise on its own. Recording and enforcing are separate
    so a caller can always account for a call that already happened -- the
    tokens were spent whether or not the budget allows them, and a ledger that
    refuses to record the call that broke the budget loses exactly the entry
    you most want to see.
    """

    prices: Mapping[str, ModelPrice]
    budget_usd: float | None = None
    entries: list[LedgerEntry] = field(default_factory=list)
    # An unpriced model is an error by default. Silently costing it at zero
    # produces a ledger that reads as under budget while real money is spent,
    # which is worse than no ledger.
    strict: bool = True

    def price_for(self, model: str) -> ModelPrice | None:
        price = self.prices.get(model)
        if price is None and self.strict:
            raise KeyError(
                f"no price for model {model!r}; known: "
                f"{sorted(self.prices)} (pass strict=False to cost it at zero)"
            )
        return price

    def record(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        """Account for one completed call. Returns that call's cost in USD."""
        if prompt_tokens < 0 or completion_tokens < 0 or cached_tokens < 0:
            raise ValueError("token counts must be non-negative")
        price = self.price_for(model)
        cost = (
            Decimal(0)
            if price is None
            else price.cost_usd(prompt_tokens, completion_tokens, cached_tokens)
        )
        self.entries.append(
            LedgerEntry(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
            )
        )
        return cost

    def record_usage(self, model: str, usage: object) -> Decimal:
        """Record from anything exposing the `Usage` field names.

        Duck-typed rather than importing `transport.Usage` so `cost` stays
        free of a dependency on the HTTP layer: the ledger is useful for
        estimating spend from a token count that never touched a socket.
        """
        return self.record(
            model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cached_tokens=getattr(usage, "cached_tokens", 0),
        )

    @property
    def total_usd(self) -> float:
        return float(self.total_decimal)

    @property
    def total_decimal(self) -> Decimal:
        """Exact total. Prefer this over `total_usd` when comparing money."""
        return sum((e.cost_usd for e in self.entries), Decimal(0))

    @property
    def remaining_usd(self) -> float | None:
        if self.budget_usd is None:
            return None
        return float(Decimal(str(self.budget_usd)) - self.total_decimal)

    def check_budget(self) -> None:
        """Raise `BudgetExceeded` if spend has reached the budget.

        Called after each `record()` in a loop, this is what turns a runaway
        into a bounded loss. With no budget set it is a no-op, so callers can
        always call it unconditionally.
        """
        if self.budget_usd is None:
            return
        budget = Decimal(str(self.budget_usd))
        if self.total_decimal > budget:
            raise BudgetExceeded(float(self.total_decimal), float(budget))

    def would_exceed(self, cost_usd: float | Decimal) -> bool:
        """Would spending `cost_usd` more push past the budget?

        The pre-flight counterpart to `check_budget`. Cheap to call before a
        request when you can estimate its cost, and the only way to avoid
        paying for the call that breaks the budget.
        """
        if self.budget_usd is None:
            return False
        extra = cost_usd if isinstance(cost_usd, Decimal) else Decimal(str(cost_usd))
        return self.total_decimal + extra > Decimal(str(self.budget_usd))

    def by_model(self) -> dict[str, float]:
        """Spend attributed per model, for the 'what cost that much' question."""
        out: dict[str, Decimal] = {}
        for e in self.entries:
            out[e.model] = out.get(e.model, Decimal(0)) + e.cost_usd
        return {k: float(v) for k, v in sorted(out.items())}

    def summary(self) -> dict[str, object]:
        return {
            "calls": len(self.entries),
            "prompt_tokens": sum(e.prompt_tokens for e in self.entries),
            "completion_tokens": sum(e.completion_tokens for e in self.entries),
            "cached_tokens": sum(e.cached_tokens for e in self.entries),
            "total_usd": round(self.total_usd, 6),
            "budget_usd": self.budget_usd,
            "by_model": self.by_model(),
        }


def prices_from_mapping(raw: Mapping[str, Mapping[str, float]]) -> dict[str, ModelPrice]:
    """Build a price table from plain JSON-shaped data.

    Lets a price table live in a config file next to the deployment rather
    than in code, which is what makes repricing a data change.
    """
    out: dict[str, ModelPrice] = {}
    for model, spec in raw.items():
        out[model] = ModelPrice(
            input_per_mtok=float(spec["input_per_mtok"]),
            output_per_mtok=float(spec["output_per_mtok"]),
            cached_input_per_mtok=(
                float(spec["cached_input_per_mtok"])
                if "cached_input_per_mtok" in spec
                else None
            ),
        )
    return out


def total_tokens(entries: Iterable[LedgerEntry]) -> int:
    return sum(e.prompt_tokens + e.completion_tokens for e in entries)
