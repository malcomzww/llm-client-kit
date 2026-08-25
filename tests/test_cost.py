"""Cost ledger tests.

These pin the claims the module makes, not its internals: that a budget
actually terminates a loop, that cached tokens are cheaper than fresh ones,
that money does not drift, and that an unpriced model is loud rather than
free.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm_client_kit.cost import (
    BudgetExceeded,
    CostLedger,
    ModelPrice,
    prices_from_mapping,
)

# Illustrative rates. Deliberately not any real provider's price list: this
# repo commits no number it did not generate, and a hardcoded real price would
# be a stale claim about someone else's product.
PRICES = {
    "small": ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0),
    "cached": ModelPrice(
        input_per_mtok=1.0, output_per_mtok=2.0, cached_input_per_mtok=0.1
    ),
}


# --- the load-bearing claim: a budget stops a runaway ------------------


def test_budget_raises_once_spend_passes_it():
    ledger = CostLedger(PRICES, budget_usd=0.000_002)
    ledger.record("small", prompt_tokens=1, completion_tokens=1)  # $0.000003
    with pytest.raises(BudgetExceeded):
        ledger.check_budget()


def test_a_runaway_loop_terminates_itself():
    """The reason this module exists: an unbounded loop must stop on its own."""
    ledger = CostLedger(PRICES, budget_usd=0.001)
    calls = 0
    with pytest.raises(BudgetExceeded):
        while calls < 10_000:  # would run forever without the budget
            ledger.record("small", prompt_tokens=100, completion_tokens=100)
            ledger.check_budget()
            calls += 1
    assert calls < 10_000, "budget failed to terminate the loop"
    assert ledger.total_usd > 0.001


def test_check_budget_is_a_noop_without_a_budget():
    ledger = CostLedger(PRICES)
    for _ in range(100):
        ledger.record("small", prompt_tokens=10_000, completion_tokens=10_000)
        ledger.check_budget()  # must not raise
    assert ledger.remaining_usd is None


def test_exactly_at_budget_is_not_exceeded():
    """Boundary: the budget is a ceiling you may reach, not one you may pass."""
    ledger = CostLedger(PRICES, budget_usd=0.000_003)
    ledger.record("small", prompt_tokens=1, completion_tokens=1)
    ledger.check_budget()
    assert ledger.total_usd == pytest.approx(0.000_003)


def test_the_call_that_broke_the_budget_is_still_recorded():
    """Recording and enforcing are separate on purpose.

    The tokens were spent whether or not the budget allowed them. A ledger
    that refuses the entry loses exactly the call you most want to inspect.
    """
    ledger = CostLedger(PRICES, budget_usd=0.000_001)
    ledger.record("small", prompt_tokens=1000, completion_tokens=1000)
    with pytest.raises(BudgetExceeded):
        ledger.check_budget()
    assert len(ledger.entries) == 1
    assert ledger.entries[0].prompt_tokens == 1000


def test_would_exceed_predicts_without_spending():
    ledger = CostLedger(PRICES, budget_usd=0.01)
    assert not ledger.would_exceed(0.005)
    assert ledger.would_exceed(0.011)
    assert ledger.total_usd == 0.0, "would_exceed must not record anything"


# --- pricing arithmetic -------------------------------------------------


def test_cost_splits_input_and_output_rates():
    ledger = CostLedger(PRICES)
    cost = ledger.record("small", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == Decimal("3")  # 1.0 in + 2.0 out


def test_cached_tokens_are_billed_at_the_cached_rate():
    """The claim that makes prefix-cache measurement meaningful."""
    ledger = CostLedger(PRICES)
    fresh = ledger.record("cached", prompt_tokens=1_000_000, completion_tokens=0)
    discounted = ledger.record(
        "cached", prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000
    )
    assert fresh == Decimal("1")
    assert discounted == Decimal("0.1")
    assert discounted < fresh


def test_cached_rate_defaults_to_the_full_input_rate():
    """Never assume a discount the caller did not state.

    Understating spend is the one direction a cost tool must not err in.
    """
    ledger = CostLedger(PRICES)
    with_cache = ledger.record(
        "small", prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000
    )
    assert with_cache == Decimal("1")


def test_cached_tokens_cannot_exceed_prompt_tokens():
    ledger = CostLedger(PRICES)
    with pytest.raises(ValueError):
        ledger.record("small", prompt_tokens=10, completion_tokens=0, cached_tokens=11)


def test_negative_token_counts_are_rejected():
    ledger = CostLedger(PRICES)
    with pytest.raises(ValueError):
        ledger.record("small", prompt_tokens=-1, completion_tokens=0)


def test_money_does_not_drift_over_many_small_calls():
    """0.1 + 0.2 != 0.3 in binary floating point.

    Ten thousand calls at a rate chosen to be unrepresentable in binary must
    still total exactly, or the ledger is not an accounting record.
    """
    prices = {"m": ModelPrice(input_per_mtok=0.3, output_per_mtok=0.0)}
    ledger = CostLedger(prices)
    for _ in range(10_000):
        ledger.record("m", prompt_tokens=1, completion_tokens=0)
    assert ledger.total_decimal == Decimal("0.3") * Decimal(10_000) / Decimal(1_000_000)


# --- unpriced models ----------------------------------------------------


def test_unknown_model_raises_by_default():
    """Costing an unpriced model at zero produces a ledger that reads as under
    budget while real money is spent, which is worse than no ledger."""
    ledger = CostLedger(PRICES)
    with pytest.raises(KeyError):
        ledger.record("not-in-the-table", prompt_tokens=10, completion_tokens=10)


def test_unknown_model_costs_zero_when_strict_is_off():
    ledger = CostLedger(PRICES, strict=False)
    assert ledger.record("mystery", prompt_tokens=10, completion_tokens=10) == Decimal(0)
    assert len(ledger.entries) == 1


# --- attribution --------------------------------------------------------


def test_spend_is_attributable_per_model():
    """A scalar total cannot answer 'which model cost that much'."""
    ledger = CostLedger(PRICES)
    ledger.record("small", prompt_tokens=1_000_000, completion_tokens=0)
    ledger.record("cached", prompt_tokens=2_000_000, completion_tokens=0)
    by_model = ledger.by_model()
    assert by_model == {"cached": pytest.approx(2.0), "small": pytest.approx(1.0)}


def test_summary_reports_calls_tokens_and_total():
    ledger = CostLedger(PRICES, budget_usd=5.0)
    ledger.record("small", prompt_tokens=100, completion_tokens=50, cached_tokens=20)
    s = ledger.summary()
    assert s["calls"] == 1
    assert s["prompt_tokens"] == 100
    assert s["completion_tokens"] == 50
    assert s["cached_tokens"] == 20
    assert s["budget_usd"] == 5.0


def test_record_usage_accepts_any_object_with_the_usage_fields():
    """Duck-typed so the ledger does not depend on the HTTP layer."""

    class FakeUsage:
        prompt_tokens = 1_000_000
        completion_tokens = 0
        cached_tokens = 0

    ledger = CostLedger(PRICES)
    assert ledger.record_usage("small", FakeUsage()) == Decimal("1")


def test_prices_load_from_plain_data():
    """Repricing should be a config change, not a code change."""
    table = prices_from_mapping(
        {
            "m": {
                "input_per_mtok": 1.5,
                "output_per_mtok": 3.0,
                "cached_input_per_mtok": 0.15,
            }
        }
    )
    assert table["m"].cached_input_per_mtok == 0.15
    ledger = CostLedger(table)
    assert ledger.record("m", prompt_tokens=1_000_000, completion_tokens=0) == Decimal("1.5")
