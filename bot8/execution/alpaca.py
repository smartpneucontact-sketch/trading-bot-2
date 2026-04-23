"""Alpaca paper-trading execution adapter.

Turns a `{symbol: target_weight}` dict into placed orders on Alpaca, tracks
fills, and reconciles the bot's intended portfolio with what actually got
filled. Design goals:

- **Deterministic & auditable** — compute_orders() is a pure function;
  given identical inputs it always produces identical orders. Caller
  saves the output to the trade journal *before* submitting, so we have
  a full record even if the API fails mid-submit.
- **Fractional-aware** — handles both fractionable (AAPL, NVDA, most
  S&P 500) and non-fractionable (some BDCs, recent IPOs) assets.
  Non-fractionable shorts + longs are rounded to whole shares.
- **Shortable-aware** — silently drops short targets for non-shortable
  names rather than erroring out. Long-only strategies work unchanged.
- **Idempotent rebalance** — running rebalance() twice in the same day
  is safe; the second call sees positions already at target and submits
  nothing.
- **Paper-only default** — `paper=True` (the default) uses paper keys.
  Live trading requires an explicit opt-in to prevent accidents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from enum import Enum
from typing import Any

from loguru import logger

from bot8.config import get_settings


# Minimum trade size to bother placing — below this the slippage eats the alpha.
MIN_TRADE_USD = Decimal("5.00")

# How long we wait for a batch of market orders to fill before giving up.
DEFAULT_FILL_TIMEOUT_S = 120

# Polling interval for order status during wait_for_fills.
FILL_POLL_INTERVAL_S = 2.0


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Subset of Alpaca asset metadata we need for order construction."""
    symbol: str
    is_tradable: bool
    is_shortable: bool
    is_fractionable: bool


@dataclass(frozen=True, slots=True)
class OrderPlan:
    """Pure description of one order, produced before submission."""
    symbol: str
    side: OrderSide
    qty: Decimal          # share count (may be fractional)
    notional: Decimal     # signed dollar delta (long/short sign preserved)
    current_qty: Decimal
    target_qty: Decimal
    price: Decimal
    reason: str           # "new_long" | "close_short" | "rebalance_up" | ...


@dataclass(slots=True)
class RebalanceReport:
    """Result of one rebalance pass — used for the trade journal + dashboard."""
    run_id: str
    nav_before: Decimal
    planned_orders: list[OrderPlan] = field(default_factory=list)
    submitted_order_ids: list[str] = field(default_factory=list)
    filled_order_ids: list[str] = field(default_factory=list)
    rejected_order_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def n_filled(self) -> int:
        return len(self.filled_order_ids)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected_order_ids)

    @property
    def n_planned(self) -> int:
        return len(self.planned_orders)


# ---------------------------------------------------------------------------
# Pure helpers — no API calls, easy to test
# ---------------------------------------------------------------------------


def _round_qty(raw_qty: Decimal, fractionable: bool) -> Decimal:
    """Round a signed share quantity. Non-fractional names round toward zero."""
    if fractionable:
        # Alpaca supports up to 9 decimal places; 4 is plenty for equities.
        return raw_qty.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    # Whole shares only, toward zero (so we never exceed the dollar target).
    sign = 1 if raw_qty >= 0 else -1
    return Decimal(int(abs(raw_qty))) * sign


def _classify_action(current_qty: Decimal, target_qty: Decimal) -> str:
    """Human-readable reason code for the trade journal."""
    if current_qty == 0 and target_qty > 0:
        return "new_long"
    if current_qty == 0 and target_qty < 0:
        return "new_short"
    if current_qty > 0 and target_qty <= 0:
        return "close_long" if target_qty == 0 else "flip_long_to_short"
    if current_qty < 0 and target_qty >= 0:
        return "close_short" if target_qty == 0 else "flip_short_to_long"
    if abs(target_qty) > abs(current_qty):
        return "rebalance_up"
    return "rebalance_down"


def compute_orders(
    target_weights: dict[str, float],
    current_positions: dict[str, Decimal],
    prices: dict[str, Decimal],
    nav: Decimal,
    asset_info: dict[str, AssetInfo],
    min_trade_usd: Decimal = MIN_TRADE_USD,
) -> list[OrderPlan]:
    """Produce the exact list of orders to bring the portfolio to the target.

    Pure function — no side effects, fully deterministic.

    Inputs:
      target_weights     {symbol: weight}  weights sum to ~0 for long/short
      current_positions  {symbol: qty}     from Alpaca get_all_positions
      prices             {symbol: last}    recent quote for sizing
      nav                account equity in USD
      asset_info         {symbol: AssetInfo}  shortable/fractionable flags

    Returns:
      List[OrderPlan] — only includes orders ≥ min_trade_usd.
      Close orders (target_weight == 0 for a current position) are always
      included regardless of size, since leaving a stale position in the book
      contaminates future rebalances.
    """
    # Union of target + current symbols — we must consider closing stale positions
    all_symbols = set(target_weights) | set(current_positions)
    plans: list[OrderPlan] = []

    for sym in sorted(all_symbols):
        target_w = float(target_weights.get(sym, 0.0))
        current_qty = current_positions.get(sym, Decimal("0"))
        info = asset_info.get(sym)
        price = prices.get(sym)

        if info is None or price is None or price <= 0:
            # No asset metadata or price — skip (don't error; live data is best-effort)
            continue
        if not info.is_tradable:
            continue

        # Silently skip shorts on non-shortables (leave long exposure alone)
        if target_w < 0 and not info.is_shortable:
            target_w = 0.0

        target_notional = Decimal(str(target_w)) * nav
        target_qty_raw = target_notional / price
        target_qty = _round_qty(target_qty_raw, info.is_fractionable)

        diff_qty = target_qty - current_qty
        diff_notional = diff_qty * price

        # Skip trivially small trades unless we're fully closing a position
        is_full_close = (target_qty == 0 and current_qty != 0)
        if not is_full_close and abs(diff_notional) < min_trade_usd:
            continue
        if diff_qty == 0:
            continue

        side = OrderSide.BUY if diff_qty > 0 else OrderSide.SELL
        plans.append(OrderPlan(
            symbol=sym,
            side=side,
            qty=abs(diff_qty),
            notional=diff_notional,
            current_qty=current_qty,
            target_qty=target_qty,
            price=price,
            reason=_classify_action(current_qty, target_qty),
        ))

    return plans


# ---------------------------------------------------------------------------
# Alpaca adapter — wraps the SDK
# ---------------------------------------------------------------------------


class AlpacaExecutor:
    """Thin wrapper over alpaca-py for the execution primitives we need."""

    def __init__(self, paper: bool = True):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        s = get_settings()
        key = s.alpaca_api_key.get_secret_value()
        secret = s.alpaca_secret_key.get_secret_value()
        if not key or not secret:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env — "
                "cannot instantiate AlpacaExecutor"
            )

        self.trading = TradingClient(key, secret, paper=paper)
        self.data = StockHistoricalDataClient(key, secret)
        self.paper = paper
        logger.info("AlpacaExecutor initialised (paper={})", paper)

    # -- Read state ---------------------------------------------------------

    def get_nav(self) -> Decimal:
        """Account equity in USD."""
        acct = self.trading.get_account()
        return Decimal(str(acct.equity))

    def get_positions(self) -> dict[str, Decimal]:
        """Current holdings as {symbol: signed qty}."""
        positions = self.trading.get_all_positions()
        return {p.symbol: Decimal(str(p.qty)) for p in positions}

    def get_asset_info(self, symbols: list[str]) -> dict[str, AssetInfo]:
        """Fractionable/shortable/tradable flags per symbol."""
        result: dict[str, AssetInfo] = {}
        for sym in symbols:
            try:
                a = self.trading.get_asset(sym)
                result[sym] = AssetInfo(
                    symbol=a.symbol,
                    is_tradable=bool(a.tradable),
                    is_shortable=bool(a.shortable),
                    is_fractionable=bool(a.fractionable),
                )
            except Exception as e:
                logger.warning("Could not fetch asset info for {}: {}", sym, e)
        return result

    def get_latest_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        """Latest trade prices — used for order sizing."""
        from alpaca.data.requests import StockLatestQuoteRequest
        if not symbols:
            return {}
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed="iex")
        quotes = self.data.get_stock_latest_quote(req)
        out: dict[str, Decimal] = {}
        for sym, q in quotes.items():
            # Use midpoint of bid/ask when available; fall back to ask.
            bid = Decimal(str(q.bid_price or 0))
            ask = Decimal(str(q.ask_price or 0))
            if bid > 0 and ask > 0:
                out[sym] = (bid + ask) / 2
            elif ask > 0:
                out[sym] = ask
        return out

    # -- Write orders -------------------------------------------------------

    def submit_orders(self, plans: list[OrderPlan]) -> list[tuple[OrderPlan, str | None, str | None]]:
        """Submit each planned order. Returns [(plan, order_id, error_msg)]."""
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        results: list[tuple[OrderPlan, str | None, str | None]] = []
        for plan in plans:
            try:
                req = MarketOrderRequest(
                    symbol=plan.symbol,
                    qty=float(plan.qty),
                    side=AlpacaSide.BUY if plan.side == OrderSide.BUY else AlpacaSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading.submit_order(order_data=req)
                results.append((plan, str(order.id), None))
                logger.info("  submitted {} {} {} qty={} id={}",
                            plan.reason, plan.side.value, plan.symbol, plan.qty, order.id)
            except Exception as e:
                err = str(e)
                logger.warning("  REJECTED {} {} qty={}: {}",
                               plan.side.value, plan.symbol, plan.qty, err)
                results.append((plan, None, err))
        return results

    def cancel_all_orders(self) -> int:
        """Cancel any open orders. Returns count canceled."""
        try:
            cancel_responses = self.trading.cancel_orders()
            n = len(cancel_responses) if cancel_responses else 0
            logger.info("Canceled {} open orders", n)
            return n
        except Exception as e:
            logger.warning("cancel_all_orders failed: {}", e)
            return 0

    def wait_for_fills(
        self,
        order_ids: list[str],
        timeout_s: int = DEFAULT_FILL_TIMEOUT_S,
    ) -> tuple[list[str], list[str]]:
        """Poll order statuses until all terminal or timeout.

        Returns (filled_ids, other_ids). 'other' includes rejected, canceled,
        partially-filled-then-day-ended, and still-open-at-timeout.
        """
        if not order_ids:
            return [], []

        pending = set(order_ids)
        filled: list[str] = []
        rejected: list[str] = []
        start = time.monotonic()

        logger.info("Waiting for fills on {} orders (timeout {}s)", len(order_ids), timeout_s)
        while pending and (time.monotonic() - start) < timeout_s:
            for oid in list(pending):
                try:
                    order = self.trading.get_order_by_id(oid)
                    status = str(order.status).lower()
                    if status == "filled":
                        filled.append(oid)
                        pending.discard(oid)
                    elif status in {"canceled", "rejected", "expired"}:
                        rejected.append(oid)
                        pending.discard(oid)
                    # else: still "pending_new", "accepted", "partially_filled"
                except Exception as e:
                    logger.warning("get_order_by_id({}) failed: {}", oid, e)

            if pending:
                time.sleep(FILL_POLL_INTERVAL_S)

        # Anything still pending at timeout joins the "other" bucket.
        rejected.extend(pending)
        logger.info("Fills: {} filled, {} unfilled-or-rejected",
                    len(filled), len(rejected))
        return filled, rejected

    # -- High-level orchestration ------------------------------------------

    def rebalance(
        self,
        target_weights: dict[str, float],
        run_id: str,
        cancel_open_first: bool = True,
        wait_for_fills: bool = True,
    ) -> RebalanceReport:
        """End-to-end rebalance: cancel stale orders → compute plan → submit
        → wait for fills → report."""
        t0 = time.monotonic()

        if cancel_open_first:
            self.cancel_all_orders()

        nav = self.get_nav()
        current = self.get_positions()
        symbols = list(set(target_weights) | set(current))
        info = self.get_asset_info(symbols)
        prices = self.get_latest_prices(symbols)

        plans = compute_orders(
            target_weights=target_weights,
            current_positions=current,
            prices=prices,
            nav=nav,
            asset_info=info,
        )

        report = RebalanceReport(
            run_id=run_id,
            nav_before=nav,
            planned_orders=plans,
        )

        if not plans:
            logger.info("Rebalance {}: nothing to do (portfolio already at target)", run_id)
            report.duration_s = time.monotonic() - t0
            return report

        logger.info("Rebalance {}: submitting {} orders", run_id, len(plans))
        submissions = self.submit_orders(plans)
        for plan, oid, err in submissions:
            if oid:
                report.submitted_order_ids.append(oid)
            if err:
                report.errors.append(f"{plan.symbol}: {err}")

        if wait_for_fills and report.submitted_order_ids:
            filled, rejected = self.wait_for_fills(report.submitted_order_ids)
            report.filled_order_ids = filled
            report.rejected_order_ids = rejected

        report.duration_s = time.monotonic() - t0
        logger.info(
            "Rebalance {} done in {:.1f}s: planned={} submitted={} filled={} errored={}",
            run_id, report.duration_s, report.n_planned,
            len(report.submitted_order_ids), report.n_filled, len(report.errors),
        )
        return report
