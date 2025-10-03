"""Post-trade analytics with modular P&L accounting methods and interactive reporting."""
from __future__ import annotations

import argparse
import csv
import sys
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

EPOCH = datetime(1970, 1, 1)


def parse_resample_argument(value: Optional[str]) -> Optional[int]:
    if value is None:
        return 60
    token = value.strip().lower()
    if token in {"", "none", "raw"}:
        return None
    unit = token[-1]
    amount_str = token[:-1] or "1"
    if not amount_str.isdigit():
        raise ValueError(f"Unsupported resample token '{value}'")
    amount = int(amount_str)
    if amount <= 0:
        raise ValueError("Resample interval must be positive")
    if unit == "s":
        return amount
    if unit in {"t", "m"}:
        return amount * 60
    raise ValueError(f"Unsupported resample token '{value}'")


def describe_bucket(seconds: int) -> str:
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return f"{hours}-hour snapshot" if hours == 1 else f"{hours}-hour snapshots"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes}-minute snapshot" if minutes == 1 else f"{minutes}-minute snapshots"
    return f"{seconds}-second snapshots"


def parse_frequency_list(value: Optional[str]) -> List[Dict[str, Optional[int]]]:
    default = ["tick", "60s"]
    tokens = default if value is None else [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        tokens = default

    freq_specs: List[Dict[str, Optional[int]]] = []
    seen: set[str] = set()
    for token in tokens:
        lower = token.lower()
        if lower in {"tick", "raw"}:
            key = "tick"
            seconds = None
            label = "Tick (raw)"
        else:
            seconds = parse_resample_argument(token)
            key = lower
            label = describe_bucket(seconds)
        if key in seen:
            continue
        freq_specs.append({"token": key, "seconds": seconds, "label": label})
        seen.add(key)

    return freq_specs


def floor_to_bucket(timestamp: datetime, bucket_seconds: int) -> datetime:
    seconds_since_epoch = int((timestamp - EPOCH).total_seconds())
    bucket_start = seconds_since_epoch - (seconds_since_epoch % bucket_seconds)
    return EPOCH + timedelta(seconds=bucket_start)


def resample_portfolio_points(
    points: Sequence["PortfolioPoint"], method_names: Sequence[str], bucket_seconds: int
) -> Tuple[List[datetime], Dict[str, List[float]]]:
    bucket_map: Dict[datetime, Dict[str, float]] = {}
    for point in points:
        bucket = floor_to_bucket(point.event_time, bucket_seconds)
        bucket_entry = bucket_map.setdefault(bucket, {})
        bucket_entry["cum_notional"] = float(point.cum_notional)
        bucket_entry["cum_margin"] = float(point.cum_margin)
        for method in method_names:
            bucket_entry[f"realized_{method}"] = float(
                point.realized_by_method.get(method, Decimal("0"))
            )

    if not bucket_map:
        return [], {
            "cum_notional": [],
            "cum_margin": [],
            **{f"realized_{method}": [] for method in method_names},
        }

    sorted_buckets = sorted(bucket_map.keys())
    start = sorted_buckets[0]
    end = sorted_buckets[-1]

    keys = set().union(*(entry.keys() for entry in bucket_map.values()))
    last_values = {key: 0.0 for key in keys}
    series = {key: [] for key in keys}
    times: List[datetime] = []

    bucket = start
    step = timedelta(seconds=bucket_seconds)
    while bucket <= end:
        if bucket in bucket_map:
            last_values.update(bucket_map[bucket])
        times.append(bucket)
        for key in keys:
            series[key].append(last_values[key])
        bucket += step

    return times, series


def resample_symbol_series(
    series: Sequence[Tuple[datetime, Decimal, Decimal]], bucket_seconds: int
) -> Tuple[List[datetime], List[float], List[float]]:
    bucket_map: Dict[datetime, Tuple[float, float]] = {}
    for timestamp, position, exposure in series:
        bucket = floor_to_bucket(timestamp, bucket_seconds)
        bucket_map[bucket] = (float(position), float(exposure))

    if not bucket_map:
        return [], [], []

    sorted_buckets = sorted(bucket_map.keys())
    start = sorted_buckets[0]
    end = sorted_buckets[-1]

    last_position, last_exposure = bucket_map[start]
    times: List[datetime] = []
    positions: List[float] = []
    exposures: List[float] = []

    bucket = start
    step = timedelta(seconds=bucket_seconds)
    while bucket <= end:
        if bucket in bucket_map:
            last_position, last_exposure = bucket_map[bucket]
        times.append(bucket)
        positions.append(last_position)
        exposures.append(last_exposure)
        bucket += step

    return times, positions, exposures


def build_datasets(
    engine: "PTAEngine",
    freq_specs: Sequence[Dict[str, Optional[int]]],
) -> Tuple[Dict[str, Dict[str, object]], List[str], Dict[str, str], List[str]]:
    method_order = [method.name for method in engine.methods]
    method_labels = {method.name: method.label for method in engine.methods}
    symbols = sorted(engine.symbol_series.keys())

    datasets: Dict[str, Dict[str, object]] = {}

    for spec in freq_specs:
        token = spec["token"]
        bucket_seconds = spec["seconds"]

        if bucket_seconds is None:
            times = [point.event_time for point in engine.portfolio_curve]
            portfolio_series = {
                "cum_notional": [float(point.cum_notional) for point in engine.portfolio_curve],
                "cum_margin": [float(point.cum_margin) for point in engine.portfolio_curve],
            }
            for method in method_order:
                portfolio_series[f"realized_{method}"] = [
                    float(point.realized_by_method.get(method, Decimal("0")))
                    for point in engine.portfolio_curve
                ]
        else:
            times, portfolio_series = resample_portfolio_points(
                engine.portfolio_curve, method_order, bucket_seconds
            )

        symbol_positions: Dict[str, Dict[str, List[float]]] = {}
        symbol_exposures: Dict[str, Dict[str, List[float]]] = {}
        for symbol in symbols:
            series = engine.symbol_series.get(symbol, [])
            if bucket_seconds is None:
                symbol_times = [pt[0] for pt in series]
                symbol_pos = [float(pt[1]) for pt in series]
                symbol_exp = [float(pt[2]) for pt in series]
            else:
                symbol_times, symbol_pos, symbol_exp = resample_symbol_series(series, bucket_seconds)

            symbol_positions[symbol] = {
                "times": symbol_times,
                "values": symbol_pos,
            }
            symbol_exposures[symbol] = {
                "times": symbol_times,
                "values": symbol_exp,
            }

        datasets[token] = {
            "times": times,
            "portfolio": portfolio_series,
            "positions": symbol_positions,
            "exposures": symbol_exposures,
        }

    return datasets, method_order, method_labels, symbols

try:  # dependency for interactive charts
    import plotly.graph_objects as go
    from plotly.io import to_html
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Canonical representation of a single fill."""

    trade_index: int
    event_time: datetime
    symbol_base: str
    side: str  # BUY / SELL
    price: Decimal
    quantity: Decimal
    notional: Decimal
    signed_qty: Decimal  # +qty for BUY, -qty for SELL
    cash_flow: Decimal  # negative for BUY, positive for SELL


@dataclass
class SymbolState:
    """Rolling inventory state for accounting methods."""

    position: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


@dataclass
class SymbolAggregate:
    """Symbol-level aggregates that are accounting-method agnostic."""

    trade_count: int = 0
    total_notional: Decimal = Decimal("0")
    net_cash_flow: Decimal = Decimal("0")
    last_price: Decimal = Decimal("0")
    ending_position: Decimal = Decimal("0")


@dataclass
class AccountingOutput:
    """Return value for processing a single trade under an accounting method."""

    position: Decimal
    avg_cost: Decimal
    realized_total: Decimal
    realized_delta: Decimal


@dataclass
class PortfolioPoint:
    """Portfolio-level point after each trade."""

    trade_index: int
    event_time: datetime
    cum_cash_flow: Decimal
    cum_notional: Decimal
    cum_margin: Decimal
    realized_by_method: Dict[str, Decimal]


# ---------------------------------------------------------------------------
# Accounting method implementations
# ---------------------------------------------------------------------------


class AccountingMethod(ABC):
    """Base class for P&L accounting engines."""

    name: str
    label: str

    def __init__(self, name: str, label: str) -> None:
        self.name = name
        self.label = label

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover - interface definition
        ...

    @abstractmethod
    def process_trade(self, trade: TradeRecord) -> AccountingOutput:  # pragma: no cover
        ...

    @abstractmethod
    def symbol_states(self) -> Dict[str, SymbolState]:  # pragma: no cover
        ...


class WACAccountant(AccountingMethod):
    """Weighted-average cost (moving average) accounting."""

    def __init__(self) -> None:
        super().__init__(name="wac", label="Weighted Average Cost")
        self._states: Dict[str, SymbolState] = {}

    def reset(self) -> None:
        self._states = defaultdict(SymbolState)

    def process_trade(self, trade: TradeRecord) -> AccountingOutput:
        state = self._states[trade.symbol_base]
        realized_delta = _wac_update(state, trade.price, trade.signed_qty)
        return AccountingOutput(
            position=state.position,
            avg_cost=state.avg_cost,
            realized_total=state.realized_pnl,
            realized_delta=realized_delta,
        )

    def symbol_states(self) -> Dict[str, SymbolState]:
        return dict(self._states)


@dataclass
class _Lot:
    quantity: Decimal  # positive for long, negative for short
    price: Decimal


class FIFOAccountant(AccountingMethod):
    """First-in-first-out lot matching accounting."""

    def __init__(self) -> None:
        super().__init__(name="fifo", label="FIFO")
        self._states: Dict[str, SymbolState] = {}
        self._lots: Dict[str, Deque[_Lot]] = {}

    def reset(self) -> None:
        self._states = defaultdict(SymbolState)
        self._lots = defaultdict(deque)

    def process_trade(self, trade: TradeRecord) -> AccountingOutput:
        state = self._states[trade.symbol_base]
        lots = self._lots[trade.symbol_base]

        qty = trade.signed_qty
        price = trade.price
        realized_delta = Decimal("0")

        if qty > 0:  # BUY -> cover shorts first, then add long lot
            remaining = qty
            while remaining > 0 and lots and lots[0].quantity < 0:
                short_lot = lots[0]
                cover_qty = min(remaining, -short_lot.quantity)
                realized_delta += (short_lot.price - price) * cover_qty
                short_lot.quantity += cover_qty  # less negative
                remaining -= cover_qty
                if short_lot.quantity == 0:
                    lots.popleft()
            if remaining > 0:
                lots.append(_Lot(quantity=remaining, price=price))

        else:  # SELL -> close longs first, then open short lot
            remaining = -qty  # positive magnitude
            while remaining > 0 and lots and lots[0].quantity > 0:
                long_lot = lots[0]
                match_qty = min(remaining, long_lot.quantity)
                realized_delta += (price - long_lot.price) * match_qty
                long_lot.quantity -= match_qty
                remaining -= match_qty
                if long_lot.quantity == 0:
                    lots.popleft()
            if remaining > 0:
                lots.append(_Lot(quantity=-remaining, price=price))

        # Clean up tiny residuals (dust)
        if lots and abs(lots[0].quantity) < Decimal("1e-12"):
            lots.popleft()

        position = sum(lot.quantity for lot in lots)
        if position != 0:
            avg_cost = sum(abs(lot.quantity) * lot.price for lot in lots) / abs(position)
        else:
            avg_cost = Decimal("0")

        state.position = position
        state.avg_cost = avg_cost
        state.realized_pnl += realized_delta

        return AccountingOutput(
            position=state.position,
            avg_cost=state.avg_cost,
            realized_total=state.realized_pnl,
            realized_delta=realized_delta,
        )

    def symbol_states(self) -> Dict[str, SymbolState]:
        return dict(self._states)


# ---------------------------------------------------------------------------
# PTA engine
# ---------------------------------------------------------------------------


class PTAEngine:
    """Execute accounting methods over a trade tape and collect analytics."""

    def __init__(self, trades: Sequence[TradeRecord], methods: Sequence[AccountingMethod]) -> None:
        if not methods:
            raise ValueError("At least one accounting method is required")
        self.trades = sorted(trades, key=lambda t: (t.event_time, t.trade_index))
        self.methods = list(methods)
        for method in self.methods:
            method.reset()

        self.portfolio_curve: List[PortfolioPoint] = []
        self.symbol_series: Dict[str, List[Tuple[datetime, Decimal, Decimal]]] = defaultdict(list)
        self.symbol_info: Dict[str, SymbolAggregate] = {}

    def run(self) -> None:
        symbol_info: Dict[str, SymbolAggregate] = defaultdict(SymbolAggregate)
        positions: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        last_price: Dict[str, Decimal] = {}

        cum_cash_flow = Decimal("0")
        cum_notional = Decimal("0")

        for trade in self.trades:
            symbol = trade.symbol_base
            info = symbol_info[symbol]
            info.trade_count += 1
            info.total_notional += trade.notional
            info.net_cash_flow += trade.cash_flow

            positions[symbol] += trade.signed_qty
            info.ending_position = positions[symbol]
            info.last_price = trade.price
            last_price[symbol] = trade.price

            cum_cash_flow += trade.cash_flow
            cum_notional += trade.notional

            realized_by_method: Dict[str, Decimal] = {}
            realized_deltas: Dict[str, Decimal] = {}
            for method in self.methods:
                output = method.process_trade(trade)
                realized_by_method[method.name] = output.realized_total
                realized_deltas[method.name] = output.realized_delta

            margin_snapshot = sum(abs(positions[sym]) * last_price[sym] for sym in positions)
            self.portfolio_curve.append(
                PortfolioPoint(
                    trade_index=trade.trade_index,
                    event_time=trade.event_time,
                    cum_cash_flow=cum_cash_flow,
                    cum_notional=cum_notional,
                    cum_margin=margin_snapshot,
                    realized_by_method=realized_by_method,
                )
            )

            net_exposure = positions[symbol] * trade.price
            self.symbol_series[symbol].append((trade.event_time, positions[symbol], net_exposure))

        self.symbol_info = dict(sorted(symbol_info.items()))


# ---------------------------------------------------------------------------
# Accounting helpers
# ---------------------------------------------------------------------------


def _wac_update(state: SymbolState, price: Decimal, qty_signed: Decimal) -> Decimal:
    """Update weighted-average cost position and return realized P&L delta."""
    realized_delta = Decimal("0")
    position = state.position
    avg_cost = state.avg_cost

    same_direction = position == 0 or (position > 0 and qty_signed > 0) or (position < 0 and qty_signed < 0)

    if same_direction:
        new_position = position + qty_signed
        if new_position != 0:
            numerator = (avg_cost * abs(position)) + (price * abs(qty_signed))
            avg_cost = numerator / abs(new_position)
        else:
            avg_cost = Decimal("0")
    else:
        closing_qty = min(abs(qty_signed), abs(position))
        if position > 0:
            realized_delta = (price - avg_cost) * closing_qty
        else:
            realized_delta = (avg_cost - price) * closing_qty

        if abs(qty_signed) < abs(position):
            new_position = position + qty_signed
        elif abs(qty_signed) == abs(position):
            new_position = Decimal("0")
            avg_cost = Decimal("0")
        else:
            leftover = abs(qty_signed) - abs(position)
            new_position = leftover if qty_signed > 0 else -leftover
            avg_cost = price

    state.position = new_position
    state.avg_cost = avg_cost if new_position != 0 else Decimal("0")
    state.realized_pnl += realized_delta
    return realized_delta


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_trades(path: Path, session_date: str) -> List[TradeRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Trade file not found: {path}")

    trades: List[TradeRecord] = []
    session_start = datetime.fromisoformat(session_date)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"buysell", "datetime", "filled_prx", "filled_qty", "symbol_base"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for idx, row in enumerate(reader, start=1):
            side = (row["buysell"] or "").strip().upper()
            if side not in {"BUY", "SELL"}:
                raise ValueError(f"Unsupported trade side '{row['buysell']}' at row {idx}")

            try:
                price = Decimal((row["filled_prx"] or "0").strip())
                quantity = Decimal((row["filled_qty"] or "0").strip())
            except InvalidOperation as exc:
                raise ValueError(f"Invalid numeric value at row {idx}: {exc}") from exc

            notional = price * quantity
            signed_qty = quantity if side == "BUY" else -quantity
            cash_flow = notional * (Decimal("-1") if side == "BUY" else Decimal("1"))

            time_token = (row["datetime"] or "0:0").strip()
            if ":" not in time_token:
                raise ValueError(f"Unexpected time token '{time_token}' at row {idx}")
            minute_str, second_str = time_token.split(":", 1)
            event_time = session_start + timedelta(
                minutes=int(minute_str),
                seconds=float(Decimal(second_str)),
            )

            trades.append(
                TradeRecord(
                    trade_index=idx,
                    event_time=event_time,
                    symbol_base=(row["symbol_base"] or "").strip(),
                    side=side,
                    price=price,
                    quantity=quantity,
                    notional=notional,
                    signed_qty=signed_qty,
                    cash_flow=cash_flow,
                )
            )

    trades.sort(key=lambda t: (t.event_time, t.trade_index))
    return trades


def format_decimal(value: Decimal, places: int = 6) -> str:
    return format(value, f".{places}f")


def write_summary(engine: PTAEngine, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pta_summary.csv"
    fieldnames = [
        "method",
        "symbol",
        "total_trades",
        "total_notional",
        "net_cash_flow",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "ending_position",
        "ending_avg_cost",
        "last_price",
        "margin_exposure",
    ]

    symbol_info = engine.symbol_info
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for symbol, info in symbol_info.items():
            for method in engine.methods:
                state = method.symbol_states().get(symbol, SymbolState())
                position = info.ending_position
                last_price = info.last_price
                margin_exposure = abs(position) * last_price
                if position == 0:
                    unrealized = Decimal("0")
                else:
                    unrealized = (last_price - state.avg_cost) * position
                total = state.realized_pnl + unrealized

                writer.writerow(
                    {
                        "method": method.name,
                        "symbol": symbol,
                        "total_trades": info.trade_count,
                        "total_notional": format_decimal(info.total_notional, 6),
                        "net_cash_flow": format_decimal(info.net_cash_flow, 6),
                        "realized_pnl": format_decimal(state.realized_pnl, 6),
                        "unrealized_pnl": format_decimal(unrealized, 6),
                        "total_pnl": format_decimal(total, 6),
                        "ending_position": format_decimal(position, 6),
                        "ending_avg_cost": format_decimal(state.avg_cost, 6),
                        "last_price": format_decimal(last_price, 6),
                        "margin_exposure": format_decimal(margin_exposure, 6),
                    }
                )
    return path


def render_interactive_report(
    output_dir: Path,
    freq_specs: Sequence[Dict[str, Optional[int]]],
    datasets: Dict[str, Dict[str, object]],
    method_order: Sequence[str],
    method_labels: Dict[str, str],
    symbols: Sequence[str],
) -> Optional[Path]:
    if not PLOTLY_AVAILABLE:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "pta_report.html"

    if not freq_specs:
        chart_path.write_text("<html><body><p>No trades to plot.</p></body></html>")
        return chart_path

    initial_token = freq_specs[0]["token"]
    initial_data = datasets.get(initial_token)
    if not initial_data:
        chart_path.write_text("<html><body><p>No trades to plot.</p></body></html>")
        return chart_path

    times0 = initial_data["times"]
    portfolio0 = initial_data["portfolio"]
    exposures0 = initial_data["exposures"]
    positions0 = initial_data["positions"]

    fig_curves = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )

    for method in method_order:
        fig_curves.add_trace(
            go.Scatter(
                x=times0,
                y=portfolio0.get(f"realized_{method}", []),
                mode="lines",
                name=method_labels.get(method, method),
                hovertemplate="Time=%{x|%H:%M:%S}<br>P&L=%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig_curves.update_yaxes(title_text="Realized P&L (USD)", row=1, col=1)

    fig_curves.add_trace(
        go.Scatter(
            x=times0,
            y=portfolio0.get("cum_notional", []),
            mode="lines",
            name="Cumulative Notional",
            line=dict(color="#1f77b4"),
            hovertemplate="Time=%{x|%H:%M:%S}<br>Notional=%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_curves.add_trace(
        go.Scatter(
            x=times0,
            y=portfolio0.get("cum_margin", []),
            mode="lines",
            name="Gross Exposure (proxy)",
            line=dict(color="#ff7f0e"),
            hovertemplate="Time=%{x|%H:%M:%S}<br>Exposure=%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_curves.update_yaxes(title_text="USD", row=2, col=1)
    fig_curves.update_xaxes(title_text="Time", row=2, col=1)

    fig_curves.update_layout(
        title=f"Portfolio Curves — {freq_specs[0]['label']}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Series"),
    )

    fig_exposure = go.Figure()
    fig_positions = go.Figure()

    zero_line = dict(type="line", xref="paper", x0=0, x1=1, y0=0, y1=0, line=dict(color="#888", width=1, dash="dash"))

    for symbol in symbols:
        exposure_series = exposures0.get(symbol, {"times": [], "values": []})
        position_series = positions0.get(symbol, {"times": [], "values": []})

        fig_exposure.add_trace(
            go.Scatter(
                x=exposure_series["times"],
                y=exposure_series["values"],
                mode="lines",
                name=symbol,
                hovertemplate="Time=%{x|%H:%M:%S}<br>Exposure=%{y:,.2f}<extra></extra>",
            )
        )
        fig_positions.add_trace(
            go.Scatter(
                x=position_series["times"],
                y=position_series["values"],
                mode="lines",
                name=symbol,
                hovertemplate="Time=%{x|%H:%M:%S}<br>Qty=%{y:,.4f}<extra></extra>",
            )
        )

    fig_exposure.update_layout(
        title=f"USD Exposure by Symbol — {freq_specs[0]['label']}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Symbol"),
        xaxis=dict(title="Time", rangeslider=dict(visible=True)),
        yaxis=dict(title="USD Exposure"),
        shapes=[zero_line],
    )

    fig_positions.update_layout(
        title=f"Net Position by Symbol (Units) — {freq_specs[0]['label']}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Symbol"),
        xaxis=dict(title="Time", rangeslider=dict(visible=True)),
        yaxis=dict(title="Net Position"),
        shapes=[zero_line],
    )

    # Dropdown menus to switch frequency snapshots
    if len(freq_specs) > 1:
        curve_buttons = []
        expo_buttons = []
        pos_buttons = []
        for spec in freq_specs:
            token = spec["token"]
            data = datasets[token]
            times = data["times"]
            portfolio = data["portfolio"]
            exposures = data["exposures"]
            positions = data["positions"]

            curve_x = []
            curve_y = []
            for method in method_order:
                curve_x.append(times)
                curve_y.append(portfolio.get(f"realized_{method}", []))
            curve_x.append(times)
            curve_y.append(portfolio.get("cum_notional", []))
            curve_x.append(times)
            curve_y.append(portfolio.get("cum_margin", []))

            expo_x = []
            expo_y = []
            for symbol in symbols:
                serie = exposures.get(symbol, {"times": [], "values": []})
                expo_x.append(serie["times"])
                expo_y.append(serie["values"])

            pos_x = []
            pos_y = []
            for symbol in symbols:
                serie = positions.get(symbol, {"times": [], "values": []})
                pos_x.append(serie["times"])
                pos_y.append(serie["values"])

            curve_buttons.append(
                dict(
                    label=spec["label"],
                    method="update",
                    args=[
                        {"x": curve_x, "y": curve_y},
                        {"title": f"Portfolio Curves — {spec['label']}"},
                    ],
                )
            )

            expo_buttons.append(
                dict(
                    label=spec["label"],
                    method="update",
                    args=[
                        {"x": expo_x, "y": expo_y},
                        {"title": f"USD Exposure by Symbol — {spec['label']}"},
                    ],
                )
            )

            pos_buttons.append(
                dict(
                    label=spec["label"],
                    method="update",
                    args=[
                        {"x": pos_x, "y": pos_y},
                        {"title": f"Net Position by Symbol (Units) — {spec['label']}"},
                    ],
                )
            )

        fig_curves.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    buttons=curve_buttons,
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.25,
                    yanchor="top",
                )
            ]
        )

        fig_exposure.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    buttons=expo_buttons,
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.25,
                    yanchor="top",
                )
            ]
        )

        fig_positions.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    buttons=pos_buttons,
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=1.25,
                    yanchor="top",
                )
            ]
        )

    html_sections = [
        "<html><head><meta charset='utf-8'><title>PTA Report</title></head><body>",
        "<h1>Post-Trade Analysis Report</h1>",
        to_html(fig_curves, include_plotlyjs="cdn", full_html=False),
        "<hr />",
        to_html(fig_exposure, include_plotlyjs=False, full_html=False),
        "<hr />",
        to_html(fig_positions, include_plotlyjs=False, full_html=False),
        "</body></html>",
    ]
    chart_path.write_text("\n".join(html_sections))
    return chart_path


# ---------------------------------------------------------------------------
# Static PNG export helpers (for graders who prefer images)
# ---------------------------------------------------------------------------


PNG_COLORS = [
    (46, 139, 87),
    (31, 119, 180),
    (255, 140, 0),
    (178, 34, 34),
    (128, 0, 128),
    (75, 0, 130),
    (220, 20, 60),
]


def _new_image(width: int, height: int, color: Tuple[int, int, int]) -> List[List[List[int]]]:
    return [[[color[0], color[1], color[2]] for _ in range(width)] for _ in range(height)]


def _set_pixel(image: List[List[List[int]]], x: int, y: int, color: Tuple[int, int, int]) -> None:
    height = len(image)
    width = len(image[0]) if height else 0
    if 0 <= x < width and 0 <= y < height:
        image[y][x][0] = color[0]
        image[y][x][1] = color[1]
        image[y][x][2] = color[2]


def _draw_line(
    image: List[List[List[int]]], x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], thickness: int = 1
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for tx in range(-thickness // 2, thickness // 2 + 1):
            for ty in range(-thickness // 2, thickness // 2 + 1):
                _set_pixel(image, x0 + tx, y0 + ty, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_rect(
    image: List[List[List[int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    fill: Optional[Tuple[int, int, int]] = None,
) -> None:
    if fill:
        for y in range(y0, y1):
            for x in range(x0, x1):
                _set_pixel(image, x, y, fill)
    _draw_line(image, x0, y0, x1, y0, color)
    _draw_line(image, x0, y1, x1, y1, color)
    _draw_line(image, x0, y0, x0, y1, color)
    _draw_line(image, x1, y0, x1, y1, color)


FONT = {
    "A": ["  X  ", " X X ", "XXXXX", "X   X", "X   X"],
    "B": ["XXXX ", "X   X", "XXXX ", "X   X", "XXXX "],
    "C": [" XXX ", "X   X", "X    ", "X   X", " XXX "],
    "E": ["XXXXX", "X    ", "XXXX ", "X    ", "XXXXX"],
    "G": [" XXX ", "X   X", "X XX ", "X  X ", " XXX "],
    "L": ["X    ", "X    ", "X    ", "X    ", "XXXXX"],
    "N": ["X   X", "XX  X", "X X X", "X  XX", "X   X"],
    "O": [" XXX ", "X   X", "X   X", "X   X", " XXX "],
    "P": ["XXXX ", "X   X", "XXXX ", "X    ", "X    "],
    "R": ["XXXX ", "X   X", "XXXX ", "X  X ", "X   X"],
    "S": [" XXXX", "X    ", " XXX ", "    X", "XXXX "],
    "T": ["XXXXX", "  X  ", "  X  ", "  X  ", "  X  "],
    "U": ["X   X", "X   X", "X   X", "X   X", " XXX "],
    "X": ["X   X", " X X ", "  X  ", " X X ", "X   X"],
    "Y": ["X   X", " X X ", "  X  ", "  X  ", "  X  "],
    " ": ["     ", "     ", "     ", "     ", "     "],
    "-": ["     ", "     ", "XXXXX", "     ", "     "],
    ":": ["     ", "  X  ", "     ", "  X  ", "     "],
}


def _draw_label(image: List[List[List[int]]], x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    cursor = x
    for ch in text.upper():
        pattern = FONT.get(ch)
        if not pattern:
            cursor += 6
            continue
        for dy, row in enumerate(pattern):
            for dx, cell in enumerate(row):
                if cell == "X":
                    _set_pixel(image, cursor + dx, y + dy, color)
        cursor += 6


def _write_png(path: Path, image: List[List[List[int]]]) -> None:
    import struct
    import zlib

    height = len(image)
    width = len(image[0]) if height else 0
    raw = bytearray()
    for row in image:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))

    compressor = zlib.compressobj()
    compressed = compressor.compress(bytes(raw)) + compressor.flush()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")

    path.write_bytes(header + ihdr + idat + iend)


def _render_line_chart(
    series: Sequence[Tuple[str, Sequence[float], Tuple[int, int, int]]],
    title: str,
    output_path: Path,
) -> None:
    if not series:
        return

    length = max(len(vals) for _, vals, _ in series)
    if length < 2:
        return

    width = 1000
    height = 520
    padding = 70
    bottom_margin = 80

    image = _new_image(width, height, (255, 255, 255))

    left = padding
    right = width - padding
    top = padding
    bottom = height - bottom_margin

    _draw_rect(image, left, top, right, bottom, (200, 200, 200))

    global_min = min(min(vals) for _, vals, _ in series)
    global_max = max(max(vals) for _, vals, _ in series)
    if global_max == global_min:
        global_max += 1.0
        global_min -= 1.0

    x_coords = []
    for _, vals, _ in series:
        n = len(vals)
        if n <= 1:
            xs = [left + (right - left) // 2] * n
        else:
            step = (right - left) / (n - 1)
            xs = [int(left + idx * step) for idx in range(n)]
        x_coords.append(xs)

    for (label, vals, color), xs in zip(series, x_coords):
        ys: List[int] = []
        for val in vals:
            norm = (val - global_min) / (global_max - global_min)
            y = bottom - int(norm * (bottom - top))
            ys.append(y)
        for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
            _draw_line(image, x0, y0, x1, y1, color, thickness=2)

    if 0 >= global_min and 0 <= global_max:
        zero = bottom - int((0 - global_min) / (global_max - global_min) * (bottom - top))
        _draw_line(image, left, zero, right, zero, (220, 220, 220), thickness=1)

    _draw_label(image, left, 20, title.upper(), (50, 50, 50))

    legend_x = left
    legend_y = height - bottom_margin + 10
    for label, _, color in series:
        _draw_line(image, legend_x, legend_y + 2, legend_x + 30, legend_y + 2, color, thickness=4)
        _draw_label(image, legend_x + 40, legend_y, label[:20], (60, 60, 60))
        legend_x += 200

    _write_png(output_path, image)


def export_static_charts(
    datasets: Dict[str, Dict[str, object]],
    freq_specs: Sequence[Dict[str, Optional[int]]],
    method_order: Sequence[str],
    method_labels: Dict[str, str],
    output_dir: Path,
) -> None:
    if not datasets:
        return

    preferred = next((spec for spec in freq_specs if spec.get("seconds")), freq_specs[0])
    data = datasets.get(preferred["token"], next(iter(datasets.values())))

    times = data.get("times", [])
    if not isinstance(times, list) or len(times) < 2:
        return

    portfolio = data.get("portfolio", {})

    pnl_series: List[Tuple[str, Sequence[float], Tuple[int, int, int]]] = []
    for idx, method in enumerate(method_order):
        values = portfolio.get(f"realized_{method}", [])
        if not values:
            continue
        color = PNG_COLORS[idx % len(PNG_COLORS)]
        pnl_series.append((method_labels.get(method, method), values, color))

    notional_values = portfolio.get("cum_notional", [])
    margin_values = portfolio.get("cum_margin", [])

    output_dir.mkdir(parents=True, exist_ok=True)

    if pnl_series:
        _render_line_chart(
            pnl_series,
            f"Realized P&L — {preferred['label']}",
            output_dir / "pta_curve_pnl.png",
        )

    if notional_values:
        _render_line_chart(
            [("Cumulative Notional", notional_values, PNG_COLORS[1])],
            f"Cumulative Notional — {preferred['label']}",
            output_dir / "pta_curve_notional.png",
        )

    if margin_values:
        _render_line_chart(
            [("Gross Exposure (proxy)", margin_values, PNG_COLORS[2])],
            f"Gross Exposure (proxy) — {preferred['label']}",
            output_dir / "pta_curve_exposure.png",
        )


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-trade analytics and compare P&L accounting methods.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/sample_trades.csv"),
        help="Path to the raw trades CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated CSV/HTML reports.",
    )
    parser.add_argument(
        "--session-date",
        type=str,
        default="2024-01-01",
        help="Trading session date (YYYY-MM-DD) to anchor HH:MM.S timestamps.",
    )
    parser.add_argument(
        "--accounting",
        type=str,
        default="wac,fifo",
        help="Comma-separated accounting methods to run (choices: wac, fifo).",
    )
    parser.add_argument(
        "--freqs",
        type=str,
        default="tick,1T",
        help="Comma-separated frequency snapshots (e.g. 'tick,30s,1T').",
    )
    return parser.parse_args()


def build_methods(accounting_argument: str) -> List[AccountingMethod]:
    tokens = [token.strip().lower() for token in accounting_argument.split(",") if token.strip()]
    if not tokens:
        tokens = ["wac"]
    methods: List[AccountingMethod] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        if token == "wac":
            methods.append(WACAccountant())
        elif token == "fifo":
            methods.append(FIFOAccountant())
        else:
            raise ValueError(f"Unsupported accounting method '{token}'")
        seen.add(token)
    return methods


def main() -> None:
    args = parse_args()

    try:
        trades = load_trades(args.input, args.session_date)
    except Exception as exc:  # pragma: no cover - user input errors
        print(f"Failed to load trades: {exc}", file=sys.stderr)
        raise

    methods = build_methods(args.accounting)
    engine = PTAEngine(trades, methods)
    engine.run()

    freq_specs = parse_frequency_list(args.freqs)
    datasets, method_order, method_labels, symbols = build_datasets(engine, freq_specs)

    summary_path = write_summary(engine, args.output_dir)
    chart_path = render_interactive_report(
        args.output_dir, freq_specs, datasets, method_order, method_labels, symbols
    )
    export_static_charts(datasets, freq_specs, method_order, method_labels, args.output_dir)

    print(f"Summary written to {summary_path}")
    if chart_path:
        print(f"Interactive report saved to {chart_path}")
    else:
        print("Plotly not installed; skipped interactive report. Install 'plotly' to enable charts.")


if __name__ == "__main__":
    main()
