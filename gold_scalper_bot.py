"""
GOLD (XAUUSD) SCALPING BOT - MetaTrader 5 + Python
===================================================
Strategy: "Trend-Filtered EMA Momentum Scalp"
  1) TREND FILTER (M5): EMA50 vs EMA200 + price on the correct side of EMA50
  2) ENTRY (M1): EMA9 crosses EMA21 in the trend direction
  3) CONFIRMATION: RSI(14) momentum zone, ATR volatility floor, spread cap
  4) EXIT: ATR-based SL, fixed R:R take profit, move SL to break-even at 1R
  5) RISK: % risk per trade, daily loss limit, max trades/day, session filter

SETUP
  pip install MetaTrader5 pandas numpy
  - Windows only (MetaTrader5 package requirement). Open MT5, log in, enable
    "Algo Trading". Check your broker's gold symbol name (XAUUSD, GOLD, XAUUSDm...).
  - START ON A DEMO ACCOUNT with DRY_RUN = True, then DRY_RUN = False on demo,
    and only consider live after weeks of forward-testing.

WARNING: No strategy is guaranteed to be profitable. Scalping is highly sensitive
to spread, slippage and broker execution. Trading leveraged products can lose
more than you expect. This is educational code, not financial advice.
"""

import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ============================ CONFIGURATION ============================
CFG = {
    "symbol": "XAUUSD",          # change to your broker's gold symbol
    "dry_run": True,             # True = log signals only, no orders
    "magic": 20260427,

    # Timeframes
    "tf_entry": mt5.TIMEFRAME_M1,
    "tf_trend": mt5.TIMEFRAME_M5,

    # Indicators
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend_fast": 50,
    "ema_trend_slow": 200,
    "rsi_period": 14,
    "atr_period": 14,

    # Filters
    "min_atr": 0.40,             # min M1 ATR in price units (skip dead market)
    "max_spread_points": 35,     # skip when spread too wide (35 pts = $0.35 on most brokers)
    "rsi_long_range": (50, 70),
    "rsi_short_range": (30, 50),
    "session_utc": (7, 17),      # trade London + NY overlap only (UTC hours)

    # Exits
    "sl_atr_mult": 1.5,          # stop = 1.5 x ATR
    "rr": 1.5,                   # take profit = 1.5 x risk
    "breakeven_at_r": 1.0,       # move SL to entry after +1R

    # Risk management
    "risk_pct": 0.5,             # % of equity risked per trade
    "max_trades_per_day": 8,
    "max_daily_loss_pct": 2.0,   # stop trading for the day after -2%
    "max_open_positions": 1,
    "deviation": 20,             # max slippage in points
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gold_scalper.log")],
)
log = logging.getLogger("gold_scalper")


# ============================== INDICATORS =============================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def get_rates(symbol, tf, bars=400) -> pd.DataFrame | None:
    r = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if r is None or len(r) < 250:
        return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.iloc[:-1].reset_index(drop=True)  # drop the still-forming candle


# ================================ SIGNAL ===============================
def get_signal():
    """Returns ('BUY'|'SELL'|None, atr_value, last_bar_time)."""
    c = CFG
    m1 = get_rates(c["symbol"], c["tf_entry"])
    m5 = get_rates(c["symbol"], c["tf_trend"])
    if m1 is None or m5 is None:
        return None, None, None

    # Trend (M5)
    m5["e_fast"] = ema(m5["close"], c["ema_trend_fast"])
    m5["e_slow"] = ema(m5["close"], c["ema_trend_slow"])
    t = m5.iloc[-1]
    uptrend = t["e_fast"] > t["e_slow"] and t["close"] > t["e_fast"]
    downtrend = t["e_fast"] < t["e_slow"] and t["close"] < t["e_fast"]

    # Entry (M1)
    m1["f"] = ema(m1["close"], c["ema_fast"])
    m1["s"] = ema(m1["close"], c["ema_slow"])
    m1["rsi"] = rsi(m1["close"], c["rsi_period"])
    m1["atr"] = atr(m1, c["atr_period"])
    cur, prev = m1.iloc[-1], m1.iloc[-2]

    cross_up = prev["f"] <= prev["s"] and cur["f"] > cur["s"]
    cross_dn = prev["f"] >= prev["s"] and cur["f"] < cur["s"]

    if cur["atr"] < c["min_atr"]:
        return None, cur["atr"], cur["time"]

    lo, hi = c["rsi_long_range"]
    if uptrend and cross_up and lo <= cur["rsi"] <= hi:
        return "BUY", cur["atr"], cur["time"]
    lo, hi = c["rsi_short_range"]
    if downtrend and cross_dn and lo <= cur["rsi"] <= hi:
        return "SELL", cur["atr"], cur["time"]
    return None, cur["atr"], cur["time"]


# ============================= RISK / SIZING ===========================
def calc_lot(sl_distance: float) -> float:
    info = mt5.symbol_info(CFG["symbol"])
    equity = mt5.account_info().equity
    risk_money = equity * CFG["risk_pct"] / 100
    loss_per_lot = sl_distance / info.trade_tick_size * info.trade_tick_value
    if loss_per_lot <= 0:
        return info.volume_min
    lot = risk_money / loss_per_lot
    step = info.volume_step
    lot = np.floor(lot / step) * step
    lot = max(info.volume_min, min(info.volume_max, lot))
    return round(lot, 2)


def today_stats():
    """Closed P/L and trade count for today (UTC) for this bot's magic number."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    deals = mt5.history_deals_get(start, datetime.now(timezone.utc))
    pnl, trades = 0.0, 0
    for d in deals or []:
        if d.magic == CFG["magic"] and d.entry == mt5.DEAL_ENTRY_OUT:
            pnl += d.profit + d.commission + d.swap
            trades += 1
    return pnl, trades


def can_trade() -> bool:
    h = datetime.now(timezone.utc)
    if h.weekday() >= 5:
        return False
    a, b = CFG["session_utc"]
    if not (a <= h.hour < b):
        return False

    pnl, trades = today_stats()
    equity = mt5.account_info().equity
    if trades >= CFG["max_trades_per_day"]:
        return False
    if pnl <= -equity * CFG["max_daily_loss_pct"] / 100:
        log.warning("Daily loss limit hit (%.2f). Trading paused.", pnl)
        return False

    tick = mt5.symbol_info_tick(CFG["symbol"])
    info = mt5.symbol_info(CFG["symbol"])
    spread_pts = (tick.ask - tick.bid) / info.point
    if spread_pts > CFG["max_spread_points"]:
        log.info("Spread too wide: %.0f pts", spread_pts)
        return False
    return True


# ================================ ORDERS ===============================
def filling_mode():
    fm = mt5.symbol_info(CFG["symbol"]).filling_mode
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def open_trade(side: str, atr_val: float):
    sym = CFG["symbol"]
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    sl_dist = atr_val * CFG["sl_atr_mult"]
    tp_dist = sl_dist * CFG["rr"]
    digits = info.digits

    if side == "BUY":
        price, sl, tp, otype = tick.ask, tick.ask - sl_dist, tick.ask + tp_dist, mt5.ORDER_TYPE_BUY
    else:
        price, sl, tp, otype = tick.bid, tick.bid + sl_dist, tick.bid - tp_dist, mt5.ORDER_TYPE_SELL

    lot = calc_lot(sl_dist)
    log.info("SIGNAL %s | lot=%.2f price=%.2f SL=%.2f TP=%.2f ATR=%.2f",
             side, lot, price, sl, tp, atr_val)
    if CFG["dry_run"]:
        return

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": lot,
        "type": otype,
        "price": price,
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": CFG["deviation"],
        "magic": CFG["magic"],
        "comment": "gold_scalper",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode(),
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Order failed: %s", res)
    else:
        log.info("Order placed. Ticket %s", res.order)


def manage_positions():
    """Move SL to break-even once price has moved 1R in our favour."""
    sym = CFG["symbol"]
    digits = mt5.symbol_info(sym).digits
    tick = mt5.symbol_info_tick(sym)
    for p in mt5.positions_get(symbol=sym) or []:
        if p.magic != CFG["magic"] or p.tp == 0:
            continue
        one_r = abs(p.tp - p.price_open) / CFG["rr"]
        if p.type == mt5.POSITION_TYPE_BUY:
            if tick.bid - p.price_open >= one_r * CFG["breakeven_at_r"] and p.sl < p.price_open:
                new_sl = p.price_open
            else:
                continue
        else:
            if p.price_open - tick.ask >= one_r * CFG["breakeven_at_r"] and (p.sl > p.price_open or p.sl == 0):
                new_sl = p.price_open
            else:
                continue
        r = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": sym,
            "position": p.ticket,
            "sl": round(new_sl, digits),
            "tp": p.tp,
            "magic": CFG["magic"],
        })
        log.info("Break-even set on #%s -> %s", p.ticket, getattr(r, "retcode", None))


def open_positions_count() -> int:
    ps = mt5.positions_get(symbol=CFG["symbol"]) or []
    return sum(1 for p in ps if p.magic == CFG["magic"])


# ================================ MAIN LOOP ============================
def main():
    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    if not mt5.symbol_select(CFG["symbol"], True):
        raise SystemExit(f"Symbol {CFG['symbol']} not found. Check broker symbol name.")

    acc = mt5.account_info()
    log.info("Connected: account %s | balance %.2f | DRY_RUN=%s",
             acc.login, acc.balance, CFG["dry_run"])

    last_bar = None
    try:
        while True:
            manage_positions()

            sig, atr_val, bar_time = get_signal()
            # act only once per new closed M1 bar
            if bar_time is not None and bar_time != last_bar:
                last_bar = bar_time
                if sig and can_trade() and open_positions_count() < CFG["max_open_positions"]:
                    open_trade(sig, atr_val)

            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
