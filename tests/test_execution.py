"""Unit tests for compute_orders() — the pure heart of the Alpaca adapter."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot8.execution.alpaca import (
    AssetInfo,
    MIN_TRADE_USD,
    OrderPlan,
    OrderSide,
    _round_qty,
    compute_orders,
)


def _info(sym: str, tradable=True, shortable=True, fractionable=True) -> AssetInfo:
    return AssetInfo(
        symbol=sym,
        is_tradable=tradable,
        is_shortable=shortable,
        is_fractionable=fractionable,
    )


class TestRoundQty:
    def test_fractional(self) -> None:
        assert _round_qty(Decimal("1.23456789"), fractionable=True) == Decimal("1.2345")

    def test_whole_shares_rounds_toward_zero(self) -> None:
        assert _round_qty(Decimal("3.9"), fractionable=False) == Decimal("3")
        assert _round_qty(Decimal("-3.9"), fractionable=False) == Decimal("-3")

    def test_zero_stays_zero(self) -> None:
        assert _round_qty(Decimal("0"), fractionable=True) == Decimal("0")
        assert _round_qty(Decimal("0"), fractionable=False) == Decimal("0")


class TestComputeOrders:
    """All tests use $100K NAV and $100/share for simplicity."""

    NAV = Decimal("100000")
    PRICES = {"AAPL": Decimal("100"), "MSFT": Decimal("100"), "NVDA": Decimal("100")}
    INFO = {
        "AAPL": _info("AAPL"),
        "MSFT": _info("MSFT"),
        "NVDA": _info("NVDA"),
    }

    def test_empty_inputs(self) -> None:
        assert compute_orders({}, {}, {}, self.NAV, {}) == []

    def test_new_long_position(self) -> None:
        # Target 2% long on AAPL; currently 0.
        plans = compute_orders(
            target_weights={"AAPL": 0.02},
            current_positions={},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert len(plans) == 1
        p = plans[0]
        assert p.symbol == "AAPL"
        assert p.side == OrderSide.BUY
        assert p.qty == Decimal("20.0000")  # $2000 / $100
        assert p.reason == "new_long"

    def test_new_short_position(self) -> None:
        plans = compute_orders(
            target_weights={"AAPL": -0.02},
            current_positions={},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert len(plans) == 1
        p = plans[0]
        assert p.side == OrderSide.SELL
        assert p.qty == Decimal("20.0000")
        assert p.reason == "new_short"

    def test_no_op_when_already_at_target(self) -> None:
        plans = compute_orders(
            target_weights={"AAPL": 0.02},
            current_positions={"AAPL": Decimal("20")},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert plans == []

    def test_close_position_when_dropped_from_target(self) -> None:
        # Currently long AAPL, target no longer mentions it → must close.
        plans = compute_orders(
            target_weights={},
            current_positions={"AAPL": Decimal("20")},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert len(plans) == 1
        p = plans[0]
        assert p.side == OrderSide.SELL
        assert p.qty == Decimal("20")
        assert p.reason == "close_long"

    def test_flip_long_to_short(self) -> None:
        plans = compute_orders(
            target_weights={"AAPL": -0.02},
            current_positions={"AAPL": Decimal("20")},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert len(plans) == 1
        p = plans[0]
        assert p.side == OrderSide.SELL
        # Sells 40 shares: 20 to close long + 20 to open short
        assert p.qty == Decimal("40.0000")
        assert p.reason == "flip_long_to_short"

    def test_drops_shorts_on_non_shortable(self) -> None:
        info = {**self.INFO, "AAPL": _info("AAPL", shortable=False)}
        plans = compute_orders(
            target_weights={"AAPL": -0.02},
            current_positions={},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=info,
        )
        assert plans == []  # silently dropped

    def test_drops_non_tradable(self) -> None:
        info = {**self.INFO, "AAPL": _info("AAPL", tradable=False)}
        plans = compute_orders(
            target_weights={"AAPL": 0.02},
            current_positions={},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=info,
        )
        assert plans == []

    def test_skips_trivially_small_trades(self) -> None:
        # Target 0.00001% = $1 — below $5 threshold
        plans = compute_orders(
            target_weights={"AAPL": 0.00001},
            current_positions={},
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert plans == []

    def test_always_closes_even_if_tiny(self) -> None:
        # Even a $1 position gets closed — can't leave stale positions.
        plans = compute_orders(
            target_weights={},
            current_positions={"AAPL": Decimal("0.01")},  # ~$1
            prices=self.PRICES,
            nav=self.NAV,
            asset_info=self.INFO,
        )
        assert len(plans) == 1
        assert plans[0].reason == "close_long"

    def test_whole_shares_only_for_non_fractionable(self) -> None:
        info = {**self.INFO, "AAPL": _info("AAPL", fractionable=False)}
        # $2000 at $100 = 20 shares exactly, fine
        # Make it $2050 to force rounding
        prices = {**self.PRICES, "AAPL": Decimal("102.5")}
        plans = compute_orders(
            target_weights={"AAPL": 0.02},
            current_positions={},
            prices=prices,
            nav=self.NAV,
            asset_info=info,
        )
        assert len(plans) == 1
        # $2000 / $102.5 = 19.51 → rounds down to 19
        assert plans[0].qty == Decimal("19")
