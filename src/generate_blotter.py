"""Generate a synthetic trade blotter conforming to the V9.4C event schema.

Stands in for lost paper-trading logs. The data is fabricated; the SHAPE is the
contract. When real output arrives from the paper account it populates these
same tables and nothing downstream changes.

Deterministic: seeded RNG, so the demo path and the eval harness reproduce.

    python -m src.generate_blotter --days 120 --seed 42

Models the real mechanics rather than a generic blotter:
  * Co1/Co2 + Tail (L/U), not fixed long/short roles
  * bucket-driven leg weights (W1/W2) and position multipliers, 0.0x buckets disabled
  * alpha accounting: W1*co1_ret - W2*co2_ret - beta*index_ret
  * ticker-level order aggregation across pairs, with fills allocated back to tags
  * LMT orders with 45s timeout falling back to MKT
  * stop-loss orders on the short leg, SQLOSS_{tag}, 0.40 alpha threshold

Planted scenarios (the agent needs real failures to investigate):
  * a wide-spread LMT that times out and falls back to MKT with heavy slippage
  * a stop-loss trigger leaving an orphaned long leg, then orphan closure
  * a delisting force-closing a live pair
  * a reconciliation mismatch halting the run
  * an emergency leverage breach
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, datetime, time, timedelta

from sqlalchemy import text

from src import config as cfg
from src.db import create_schema, get_engine, table_counts

UNIVERSE = {
    "VGT": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "ADBE", "AMD", "INTC", "CSCO"],
    "VFH": ["JPM", "BAC", "WFC", "GS", "MS", "SCHW", "BLK", "C", "AXP", "SPGI"],
    "VIS": ["GE", "CAT", "UNP", "HON", "BA", "LMT", "DE", "UPS", "RTX", "MMM"],
    "VHT": ["LLY", "JNJ", "UNH", "MRK", "ABBV", "TMO", "ABT", "PFE", "DHR", "AMGN"],
    "VCR": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM"],
    "VOX": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "EA", "WBD"],
}

BASE_PRICES = {
    "AAPL": 228, "MSFT": 415, "NVDA": 132, "AVGO": 172, "CRM": 265, "ORCL": 178,
    "ADBE": 385, "AMD": 142, "INTC": 22, "CSCO": 58, "JPM": 238, "BAC": 44,
    "WFC": 71, "GS": 555, "MS": 118, "SCHW": 76, "BLK": 985, "C": 68,
    "AXP": 288, "SPGI": 512, "GE": 188, "CAT": 385, "UNP": 232, "HON": 218,
    "BA": 152, "LMT": 512, "DE": 425, "UPS": 128, "RTX": 122, "MMM": 132,
    "LLY": 782, "JNJ": 152, "UNH": 552, "MRK": 98, "ABBV": 178, "TMO": 512,
    "ABT": 115, "PFE": 26, "DHR": 232, "AMGN": 285, "AMZN": 222, "TSLA": 342,
    "HD": 398, "MCD": 292, "NKE": 76, "LOW": 258, "SBUX": 98, "TJX": 122,
    "BKNG": 4850, "GM": 52, "GOOGL": 196, "META": 585, "NFLX": 745, "DIS": 96,
    "CMCSA": 38, "VZ": 42, "T": 23, "TMUS": 228, "EA": 148, "WBD": 11,
}

INDEX_PRICES = {"VGT": 625, "VFH": 122, "VIS": 268, "VHT": 268, "VCR": 348, "VOX": 178}

# Sized so a full book runs near the configured leverage band rather than
# at a tenth of it, which left every leverage check permanently passing.
BASE_TRADE_SIZE = 3400.0


def _uid(p: str) -> str:
    return f"{p}_{uuid.uuid4().hex[:10]}"


def _trading_days(end: date, count: int) -> list[date]:
    days, cur = [], end
    while len(days) < count:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    return sorted(days)


def build_price_paths(rng, calendar):
    """Precompute correlated daily price paths.

    Constituents move with their index (beta) plus an idiosyncratic component.
    This matters: independent draws would make pair legs uncorrelated and
    produce alpha an order of magnitude too large for a market-neutral book.
    """
    px = {}
    for idx, base in INDEX_PRICES.items():
        level = float(base)
        for day in calendar:
            level *= 1 + rng.gauss(0.0003, 0.009)
            px[(idx, day)] = round(level, 2)

    # Idiosyncratic component follows an Ornstein-Uhlenbeck process rather than
    # a random walk. This matters economically: with pure random walks, pair
    # legs never converge, the strategy has no edge by construction, and any
    # barrier structure produces negative expectancy. OU gives deviations a
    # half-life, which is what pair selection is actually screening for.
    kappa = 0.06          # mean-reversion speed -> ~11 day half-life
    sigma_idio = 0.011
    betas = {}
    for idx, tickers in UNIVERSE.items():
        for ticker in tickers:
            betas[ticker] = rng.uniform(0.80, 1.25)
            spread = 0.0
            level = float(BASE_PRICES[ticker])
            prev_idx = float(INDEX_PRICES[idx])
            for day in calendar:
                idx_level = px[(idx, day)]
                idx_ret = (idx_level - prev_idx) / prev_idx
                prev_idx = idx_level
                prev_spread = spread
                spread = spread * (1 - kappa) + rng.gauss(0, sigma_idio)
                level *= 1 + betas[ticker] * idx_ret + (spread - prev_spread)
                px[(ticker, day)] = round(max(level, 1.0), 2)
    return px, betas


def sum_deviation(px, co1, co2, day, calendar_index, lookback=15):
    """15-day standardised deviation of the log price ratio, as a percentile.

    Mirrors sum_deviation_15d / sum_dev_percentile in the live system: how far
    the pair's relative pricing sits from its recent norm. Computed from the
    generated price paths so that the CDF bucket genuinely describes the spread
    state — which is what makes an extreme bucket predictive of reversion.
    """
    import math
    i = calendar_index[day]
    if i < lookback:
        return None, None
    window = calendar_index["_days"][i - lookback:i + 1]
    ratios = [math.log(px[(co1, d)] / px[(co2, d)]) for d in window]
    current = ratios[-1]
    history = ratios[:-1]
    mean = sum(history) / len(history)
    var = sum((r - mean) ** 2 for r in history) / max(len(history) - 1, 1)
    sd = math.sqrt(var) if var > 0 else 1e-9
    z = (current - mean) / sd
    # Normal CDF -> percentile. Extremes sit at the tails, as the buckets expect.
    percentile = 100 * 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(current - mean, 6), round(min(max(percentile, 0.01), 99.99), 2)


def _spread_bps(rng, price, stressed=False):
    base = 2.5 if price > 100 else 7.0
    return round(base * (rng.uniform(9.0, 16.0) if stressed else rng.uniform(0.5, 2.4)), 2)


class Generator:
    def __init__(self, seed: int, days: int):
        self.rng = random.Random(seed)
        self.anchor = date.today()
        self.calendar = _trading_days(self.anchor - timedelta(days=1), days)
        self.instruments, self.runs, self.stages = [], [], []
        self.snapshots, self.evals, self.positions = [], [], []
        self.updates, self.orders, self.allocations = [], [], []
        self.fills, self.stops, self.risk_checks, self.events = [], [], [], []
        self.px, self.ticker_betas = build_price_paths(self.rng, self.calendar)
        self.cal_index = {d: i for i, d in enumerate(self.calendar)}
        self.cal_index["_days"] = self.calendar
        self.open_positions: list[dict] = []
        self.tag_counter = 0
        self.planted = set()

    # ---- setup ----------------------------------------------------------
    def build_instruments(self):
        for idx, tickers in UNIVERSE.items():
            for t in tickers:
                self.instruments.append({
                    "ticker": t, "name": t, "idx": idx, "is_active": 1,
                    "delisted_date": None, "delisting_type": None, "acquirer": None,
                })

    def _next_tag(self, idx, co1, co2, tail, day):
        self.tag_counter += 1
        return f"{idx}_{co1}_{co2}_{tail}_{day:%Y%m%d}_{self.tag_counter:03d}"

    # ---- main loop -------------------------------------------------------
    def run(self):
        self.build_instruments()
        for i, day in enumerate(self.calendar):
            self.run_day(day, i, last_week=i >= len(self.calendar) - 5)
        self._plant_delisting()
        self._plant_wide_spread_fill()

    def run_day(self, day: date, day_idx: int, last_week: bool):
        rng = self.rng
        run_id = _uid("run")
        started = datetime.combine(day, time(14, 32))
        halted = last_week and "recon" not in self.planted and day_idx == len(self.calendar) - 2

        self.runs.append({
            "run_id": run_id, "run_date": day.isoformat(),
            "started_at": started.isoformat(sep=" ", timespec="seconds"),
            "completed_at": (started + timedelta(minutes=rng.randint(6, 22)))
                .isoformat(sep=" ", timespec="seconds"),
            "outcome": "halted" if halted else "completed",
        })

        account_value = 92_000 + rng.gauss(0, 2400)
        gross = sum(p["total_notional"] for p in self.open_positions)
        leverage = round(gross / account_value, 3) if account_value else 0.0

        self.snapshots.append({
            "run_id": run_id, "snapshot_date": day.isoformat(),
            "position_count": len(self.open_positions),
            "account_value": round(account_value, 2),
            "total_gross_exposure": round(gross, 2),
            "leverage": leverage,
            "dollar_weighted_beta": round(rng.uniform(-0.12, 0.14), 4),
            "staleness_minutes": round(rng.uniform(0.2, 4.0), 2),
        })

        self._risk_checks(run_id, started, leverage, day)
        self._daily_marks(run_id, day)
        terminated = self._evaluate_terminations(run_id, day, started)
        approved = self._screen_pairs(run_id, day, started, leverage)
        executed = self._execute(run_id, day, started, approved, last_week)
        self._stages(run_id, started, terminated, approved, executed, halted, day)

        if halted:
            self.planted.add("recon")
            self.events.append({
                "event_id": _uid("evt"),
                "occurred_at": (started + timedelta(minutes=9)).isoformat(sep=" ", timespec="seconds"),
                "run_id": run_id, "event_type": "reconciliation_mismatch", "severity": "halt",
                "ticker": None, "tag": None, "order_id": None,
                "detail": (f"TWS reported {len(self.open_positions) + 1} positions against "
                           f"{len(self.open_positions)} in portfolio state. Discrepancy of 1 "
                           f"position; order submission halted pending reconciliation."),
                "remedy_action": "halt_trading", "resolved": 1, "resolution": "operator",
            })

        if rng.random() < 0.05:
            self.events.append({
                "event_id": _uid("evt"),
                "occurred_at": (started + timedelta(minutes=rng.randint(2, 18))).isoformat(sep=" ", timespec="seconds"),
                "run_id": run_id, "event_type": "ibkr_connection_error", "severity": "warning",
                "ticker": None, "tag": None, "order_id": None,
                "detail": f"IBKR gateway disconnect; retry {rng.randint(1, 3)} succeeded.",
                "remedy_action": "retry", "resolved": 1, "resolution": "auto",
            })

    # ---- risk ------------------------------------------------------------
    def _risk_checks(self, run_id, when, leverage, day):
        rng = self.rng
        breach = leverage >= cfg.EMERGENCY_LEVERAGE_THRESHOLD
        checks = [
            ("leverage_limit", None, leverage, cfg.MAX_ACCOUNT_LEVERAGE),
            ("emergency_leverage", None, leverage, cfg.EMERGENCY_LEVERAGE_THRESHOLD),
            ("portfolio_beta", None, round(rng.uniform(-0.15, 0.15), 4), 0.20),
        ]
        for idx in UNIVERSE:
            checks.append(("index_concentration", idx,
                           round(rng.uniform(4, 34), 2), cfg.MAX_INDEX_GROSS_EXPOSURE_PCT))
        for factor in cfg.FACTORS[:3]:
            checks.append(("factor_exposure", factor,
                           round(rng.uniform(-0.3, 0.3), 3), 0.35))

        for name, subject, value, threshold in checks:
            failed = abs(value) > threshold if name != "leverage_limit" else value > threshold
            self.risk_checks.append({
                "check_id": _uid("chk"), "run_id": run_id,
                "checked_at": when.isoformat(sep=" ", timespec="seconds"),
                "check_name": name, "subject": subject,
                "current_value": value, "threshold": threshold,
                "result": "Fail" if failed else "Pass",
                "action": ("halt_trading" if name.endswith("leverage") else "reject_trade") if failed else "none",
            })
            if failed and name == "emergency_leverage":
                self.events.append({
                    "event_id": _uid("evt"),
                    "occurred_at": when.isoformat(sep=" ", timespec="seconds"),
                    "run_id": run_id, "event_type": "leverage_exceeded", "severity": "halt",
                    "ticker": None, "tag": None, "order_id": None,
                    "detail": (f"Leverage {value} breached emergency threshold "
                               f"{cfg.EMERGENCY_LEVERAGE_THRESHOLD}; new entries halted."),
                    "remedy_action": "halt_trading", "resolved": 1, "resolution": "auto",
                })
        return breach

    # ---- daily marks -----------------------------------------------------
    def _daily_marks(self, run_id, day):
        rng = self.rng
        for pos in self.open_positions:
            co1_px = self.px[(pos['co1'], day)]
            co2_px = self.px[(pos['co2'], day)]
            idx_px = self.px[(pos['idx'], day)]
            co1_ret = (co1_px - pos["co1_init"]) / pos["co1_init"]
            co2_ret = (co2_px - pos["co2_init"]) / pos["co2_init"]
            idx_ret = (idx_px - pos["idx_init"]) / pos["idx_init"]
            alpha = self._alpha(pos, co1_ret, co2_ret, idx_ret)
            self.updates.append({
                "tag": pos["tag"], "update_date": day.isoformat(), "run_id": run_id,
                "live_co1_price": co1_px, "live_co2_price": co2_px, "live_index_price": idx_px,
                "co1_return_pct": round(co1_ret * 100, 4),
                "co2_return_pct": round(co2_ret * 100, 4),
                "index_return_pct": round(idx_ret * 100, 4),
                "co1_alpha_pct": round((co1_ret - pos["beta"] * idx_ret) * 100, 4),
                "co2_alpha_pct": round((co2_ret - pos["beta"] * idx_ret) * 100, 4),
                "live_alpha_return_pct": round(alpha * 100, 4),
                "days_held": (day - pos["opened"]).days,
            })
            pos["last_alpha"] = alpha
            pos["last_prices"] = (co1_px, co2_px, idx_px, co1_ret, co2_ret, idx_ret)

    @staticmethod
    def _alpha(pos, co1_ret, co2_ret, idx_ret):
        """L-tail: W1*co1 - W2*co2 - beta*index. U-tail reverses the legs."""
        if pos["tail"] == "L":
            return pos["w1"] * co1_ret - pos["w2"] * co2_ret - pos["beta"] * idx_ret
        return pos["w1"] * co2_ret - pos["w2"] * co1_ret - pos["beta"] * idx_ret

    # ---- terminations ----------------------------------------------------
    def _evaluate_terminations(self, run_id, day, when):
        rng, still_open, terminated = self.rng, [], 0
        for pos in self.open_positions:
            held = (day - pos["opened"]).days
            alpha = pos.get("last_alpha", 0.0)
            stop_hit = alpha <= -cfg.STOP_LOSS_ALPHA_THRESHOLD * pos["entry_alpha_ref"]

            if stop_hit:
                reason = "Stop Loss Triggered"
            elif alpha >= 0.018:
                reason = "Alpha Reached"
            elif held >= rng.randint(11, 26):
                reason = "Date Reached"
            elif rng.random() < 0.015:
                reason = "Earnings Alert"
            else:
                still_open.append(pos)
                continue

            self._close(run_id, pos, day, when, reason)
            terminated += 1
        self.open_positions = still_open
        return terminated

    def _close(self, run_id, pos, day, when, reason):
        rng = self.rng
        co1_px, co2_px, idx_px, co1_ret, co2_ret, idx_ret = pos.get(
            "last_prices",
            (pos["co1_init"], pos["co2_init"], pos["idx_init"], 0.0, 0.0, 0.0),
        )
        alpha = self._alpha(pos, co1_ret, co2_ret, idx_ret)

        for row in self.positions:
            if row["tag"] != pos["tag"]:
                continue
            row.update({
                "status": "closed", "exit_reason": reason,
                "termination_date": day.isoformat(),
                "holding_days": (day - pos["opened"]).days,
                "co1_at_exit": co1_px, "co2_at_exit": co2_px, "index_at_exit": idx_px,
                "co1_return_pct": round(co1_ret * 100, 4),
                "co2_return_pct": round(co2_ret * 100, 4),
                "index_return_pct": round(idx_ret * 100, 4),
                "co1_alpha_pct": round((co1_ret - pos["beta"] * idx_ret) * 100, 4),
                "co2_alpha_pct": round((co2_ret - pos["beta"] * idx_ret) * 100, 4),
                "final_alpha_return_pct": round(alpha * 100, 4),
            })
            break

        for stop in self.stops:
            if stop["tag"] == pos["tag"] and stop["status"] == "active":
                if reason == "Stop Loss Triggered":
                    stop["status"] = "triggered"
                    stop["triggered_at"] = when.isoformat(sep=" ", timespec="seconds")
                    self._orphan(run_id, pos, when)
                else:
                    stop["status"] = "cancelled"
                break

        # closing orders, one per leg
        for ticker, side, qty in (
            (pos["co1"], "SELL" if pos["tail"] == "L" else "BUY", pos["q1"]),
            (pos["co2"], "BUY" if pos["tail"] == "L" else "SELL", pos["q2"]),
        ):
            self._emit_order(run_id, day, when + timedelta(seconds=rng.randint(2, 40)),
                             ticker, side, qty,
                             [(pos["tag"], pos["pair"], qty)], stressed=False, force_mkt=True)

    def _orphan(self, run_id, pos, when):
        long_leg = pos["co1"] if pos["tail"] == "L" else pos["co2"]
        self.events.append({
            "event_id": _uid("evt"),
            "occurred_at": when.isoformat(sep=" ", timespec="seconds"),
            "run_id": run_id, "event_type": "orphan_detection", "severity": "warning",
            "ticker": long_leg, "tag": pos["tag"], "order_id": None,
            "detail": (f"Short leg stopped out on {pos['tag']}; long leg {long_leg} "
                       f"orphaned and flagged for closure."),
            "remedy_action": "close_orphan", "resolved": 1, "resolution": "auto",
        })
        self.events.append({
            "event_id": _uid("evt"),
            "occurred_at": (when + timedelta(seconds=45)).isoformat(sep=" ", timespec="seconds"),
            "run_id": run_id, "event_type": "orphan_closure", "severity": "info",
            "ticker": long_leg, "tag": pos["tag"], "order_id": None,
            "detail": f"Orphaned leg {long_leg} closed at market.",
            "remedy_action": None, "resolved": 1, "resolution": "auto",
        })

    # ---- screening -------------------------------------------------------
    def _screen_pairs(self, run_id, day, when, leverage):
        rng, approved = self.rng, []
        for _ in range(rng.randint(28, 46)):
            idx = rng.choice(list(UNIVERSE))
            co1, co2 = rng.sample(UNIVERSE[idx], 2)
            tail = rng.choice(["L", "U"])
            spread = _spread_bps(rng, min(BASE_PRICES[co1], BASE_PRICES[co2]))
            earnings = rng.randint(0, 60)
            co1_trend, co2_trend = rng.random() < 0.15, rng.random() < 0.15
            tstat = round(rng.uniform(1.1, 4.4), 3)

            fail = None
            if spread > cfg.PREFILTER_MAX_SPREAD_BPS:
                fail = "spread_hurdle"
            elif earnings <= 3:
                fail = "earnings_filter"
            elif co1_trend or co2_trend:
                fail = "trend_filter"
            elif rng.random() < 0.10:
                fail = rng.choice(["same_direction", "nominal_direction", "tstat"])

            row = {
                "eval_id": _uid("evl"), "run_id": run_id,
                "evaluated_at": when.isoformat(sep=" ", timespec="seconds"),
                "pair": f"{co1}_{co2}", "idx": idx, "co1": co1, "co2": co2, "tail": tail,
                "tstat": tstat, "weighted_spread_bps": spread, "earnings_days_out": earnings,
                "co1_trending": int(co1_trend), "co2_trending": int(co2_trend),
                "same_direction_result": "Fail" if fail == "same_direction" else "Pass",
                "nominal_direction_result": "Fail" if fail == "nominal_direction" else "Pass",
                "primary_result": "Fail" if fail else "Pass",
                "primary_fail_reason": fail,
                "volume_ratio": None, "rolling_intraday_vol": None, "iv_percentile": None,
                "volume_dominance": None, "true_last_hour_volatility": None,
                "weighted_score": None, "composite_score": None,
                "sum_deviation_15d": None, "sum_dev_percentile": None, "sum_dev_bucket": None,
                "position_multiplier": None, "is_tradeable_bucket": None,
                "shocked_factors": None, "factor_action": None,
                "spread_quality_score": None, "sum_dev_extremity_score": None,
                "composite_priority_score": None,
                "evaluation_result": None, "rejection_reason": None,
            }

            if fail:
                self.evals.append(row)
                continue

            # secondary signals + scoring
            deviation, pct = sum_deviation(self.px, co1, co2, day, self.cal_index)
            if pct is None:
                continue  # inside the lookback warm-up window
            bucket = cfg.BUCKETS[min(int(pct // 10), 9)]
            # Tail follows the direction of the deviation: if co1 is rich
            # relative to co2, short co1 and long co2.
            tail = "U" if pct > 50 else "L"
            row["tail"] = tail
            w1, w2, mult = cfg.BUCKET_SIZING[bucket]
            if tail == "U":
                w1, w2 = w2, w1
            shocked = rng.random() < 0.07
            factor = rng.choice(cfg.FACTORS) if shocked else None

            row.update({
                "volume_ratio": round(rng.uniform(0.5, 2.6), 3),
                "rolling_intraday_vol": round(rng.uniform(0.008, 0.045), 4),
                "iv_percentile": round(rng.uniform(2, 98), 2),
                "volume_dominance": round(rng.uniform(0.3, 0.9), 3),
                "true_last_hour_volatility": round(rng.uniform(0.004, 0.03), 4),
                "weighted_score": round(rng.uniform(0.2, 0.95), 4),
                "composite_score": round(rng.uniform(0.25, 0.98), 4),
                "sum_deviation_15d": deviation,
                "sum_dev_percentile": pct, "sum_dev_bucket": bucket,
                "position_multiplier": mult, "is_tradeable_bucket": int(mult > 0),
                "shocked_factors": factor, "factor_action": "SUPPRESS" if shocked else "ALLOW",
                "spread_quality_score": round(1 - spread / cfg.PREFILTER_MAX_SPREAD_BPS, 4),
                "sum_dev_extremity_score": round(abs(pct - 50) / 50, 4),
            })
            row["composite_priority_score"] = round(
                0.5 * row["composite_score"] + 0.3 * row["spread_quality_score"]
                + 0.2 * row["sum_dev_extremity_score"], 4)

            if mult == 0.0:
                reject = "bucket_not_tradeable"
            elif shocked:
                reject = "factor_shock"
            elif leverage > cfg.MAX_ACCOUNT_LEVERAGE:
                reject = "leverage"
            elif any(p["pair"] == row["pair"] for p in self.open_positions):
                reject = "duplicate"
            elif rng.random() < 0.55:
                reject = rng.choice(["max_per_ticker", "index_concentration",
                                     "position_size", "factor_exposure"])
            else:
                reject = None

            row["evaluation_result"] = "Rejected" if reject else "Approved"
            row["rejection_reason"] = reject
            self.evals.append(row)

            if not reject:
                approved.append(row)

            if shocked:
                self.events.append({
                    "event_id": _uid("evt"),
                    "occurred_at": when.isoformat(sep=" ", timespec="seconds"),
                    "run_id": run_id, "event_type": "factor_shock_exposure", "severity": "warning",
                    "ticker": None, "tag": None, "order_id": None,
                    "detail": f"{row['pair']} suppressed: {factor} factor shocked.",
                    "remedy_action": "suppress", "resolved": 1, "resolution": "auto",
                })

        return approved

    # ---- execution -------------------------------------------------------
    def _execute(self, run_id, day, when, approved, last_week):
        rng = self.rng
        approved = sorted(approved, key=lambda r: -r["composite_priority_score"])[:rng.randint(1, 7)]
        if not approved:
            return 0

        basket: dict[tuple[str, str], list] = {}
        for row in approved:
            tail = row["tail"]
            w1, w2, mult = cfg.BUCKET_SIZING[row["sum_dev_bucket"]]
            if tail == "U":
                w1, w2 = w2, w1
            notional = min(max(BASE_TRADE_SIZE * mult, cfg.MIN_POSITION_SIZE), cfg.MAX_POSITION_SIZE)
            co1_px = self.px[(row['co1'], day)]
            co2_px = self.px[(row['co2'], day)]
            idx_px = self.px[(row['idx'], day)]
            q1 = int(notional * w1 / co1_px)
            q2 = int(notional * w2 / co2_px)
            if q1 < 1 or q2 < 1 or (q1 * co1_px + q2 * co2_px) > cfg.MAX_POSITION_SIZE:
                # A single share already breaches the size cap: the name cannot
                # be traded at this notional. Recorded as a rejection rather
                # than silently rounded up past the limit.
                self.risk_checks.append({
                    "check_id": _uid("chk"), "run_id": run_id,
                    "checked_at": when.isoformat(sep=" ", timespec="seconds"),
                    "check_name": "position_size", "subject": row["pair"],
                    "current_value": round(max(co1_px, co2_px), 2),
                    "threshold": cfg.MAX_POSITION_SIZE,
                    "result": "Fail", "action": "reject_trade",
                })
                continue
            # beta in the alpha formula is the position's NET exposure to the
            # index, not a single stock's beta. For a hedged pair this is
            # W1*beta_co1 - W2*beta_co2 — near zero when weights are equal,
            # and at most ~0.2 in the skewed buckets. Using a raw ~1.0 beta
            # here over-hedges by roughly 5x and drains alpha in a rising market.
            beta = round(w1 * self.ticker_betas[row["co1"]]
                         - w2 * self.ticker_betas[row["co2"]], 4)
            tag = self._next_tag(row["idx"], row["co1"], row["co2"], tail, day)
            opened_at = when + timedelta(minutes=rng.randint(1, 30))

            self.positions.append({
                "tag": tag, "pair": row["pair"], "co1": row["co1"], "co2": row["co2"],
                "idx": row["idx"], "tail": tail, "version": "V9.2",
                "quantity1": q1, "quantity2": q2, "w1": w1, "w2": w2,
                "trade_value_co1": round(q1 * co1_px, 2), "trade_value_co2": round(q2 * co2_px, 2),
                "total_notional": round(q1 * co1_px + q2 * co2_px, 2),
                "position_multiplier": mult,
                "co1_at_initiation": co1_px, "co2_at_initiation": co2_px,
                "index_at_initiation": idx_px,
                "trade_initiation_date": opened_at.isoformat(sep=" ", timespec="seconds"),
                "entry_spread_bps": row["weighted_spread_bps"],
                "sum_dev_bucket": row["sum_dev_bucket"],
                "sum_deviation": row["sum_deviation_15d"],
                "sum_dev_percentile": row["sum_dev_percentile"],
                "weighted_score": row["weighted_score"], "composite_score": row["composite_score"],
                "beta": beta, "stop_price": None, "stop_order_id": None,
                "status": "open", "exit_reason": None, "termination_date": None,
                "holding_days": None, "co1_at_exit": None, "co2_at_exit": None,
                "index_at_exit": None, "co1_return_pct": None, "co2_return_pct": None,
                "index_return_pct": None, "co1_alpha_pct": None, "co2_alpha_pct": None,
                "final_alpha_return_pct": None,
            })

            self.open_positions.append({
                "tag": tag, "pair": row["pair"], "co1": row["co1"], "co2": row["co2"],
                "idx": row["idx"], "tail": tail, "w1": w1, "w2": w2, "beta": beta,
                "q1": q1, "q2": q2, "co1_init": co1_px, "co2_init": co2_px,
                "idx_init": idx_px, "opened": day,
                "total_notional": round(q1 * co1_px + q2 * co2_px, 2),
                "entry_alpha_ref": 0.045,
            })

            side1 = "BUY" if tail == "L" else "SELL"
            side2 = "SELL" if tail == "L" else "BUY"
            basket.setdefault((row["co1"], side1), []).append((tag, row["pair"], q1))
            basket.setdefault((row["co2"], side2), []).append((tag, row["pair"], q2))

            # stop loss on the short leg
            short_leg = row["co2"] if tail == "L" else row["co1"]
            short_px = co2_px if tail == "L" else co1_px
            short_qty = q2 if tail == "L" else q1
            stop_px = round(short_px * 1.0 + short_px * cfg.STOP_LOSS_ALPHA_THRESHOLD * 0.1, 2)
            self.stops.append({
                "stop_order_tag": f"SQLOSS_{tag}", "tag": tag, "ticker": short_leg,
                "quantity": short_qty, "entry_price": short_px, "stop_price": stop_px,
                "alpha_threshold": cfg.STOP_LOSS_ALPHA_THRESHOLD,
                "order_id": _uid("ib"),
                "placed_at": (opened_at + timedelta(seconds=30)).isoformat(sep=" ", timespec="seconds"),
                "last_updated_at": None, "status": "active", "triggered_at": None,
            })
            for p in self.positions:
                if p["tag"] == tag:
                    p["stop_price"], p["stop_order_id"] = stop_px, f"SQLOSS_{tag}"
                    break

        # aggregate by ticker+direction, then emit one order per group
        for batch_no, ((ticker, side), contribs) in enumerate(basket.items(), start=1):
            self._emit_order(run_id, day, when + timedelta(minutes=rng.randint(1, 35)),
                             ticker, side, sum(c[2] for c in contribs), contribs,
                             stressed=False, force_mkt=False, batch_number=batch_no)

        return len(approved)

    def _emit_order(self, run_id, day, when, ticker, side, shares, contribs,
                    stressed, force_mkt, batch_number=1):
        rng = self.rng
        price = self.px[(ticker, day)]
        spread = _spread_bps(rng, price, stressed)
        half = price * (spread / 10_000) / 2
        bid, ask = round(price - half, 2), round(price + half, 2)
        mid = round((bid + ask) / 2, 4)
        order_id = _uid("ord")

        # LMT unless spread fails validation, or this is a forced close
        use_mkt = force_mkt or spread > cfg.MAX_LIMIT_ORDER_SPREAD_BPS
        order_type = "MKT" if use_mkt else "LMT"
        limit_price = None
        fell_back, fallback_reason, elapsed = 0, None, round(rng.uniform(0.4, 8.0), 1)

        if use_mkt and not force_mkt:
            fallback_reason = "spread_validation_failure"
            self.events.append({
                "event_id": _uid("evt"),
                "occurred_at": when.isoformat(sep=" ", timespec="seconds"),
                "run_id": run_id, "event_type": "spread_validation_failure", "severity": "warning",
                "ticker": ticker, "tag": None, "order_id": order_id,
                "detail": (f"{ticker} spread {spread}bps exceeded "
                           f"MAX_LIMIT_ORDER_SPREAD_BPS ({cfg.MAX_LIMIT_ORDER_SPREAD_BPS}); "
                           f"routed as MKT."),
                "remedy_action": "route_mkt", "resolved": 1, "resolution": "auto",
            })
        elif not use_mkt:
            direction = 1 if side == "BUY" else -1
            limit_price = round(mid * (1 + direction * 2.0 / 10_000), 2)
            if rng.random() < 0.18:  # limit fails to fill inside the timeout
                # order_type records what was SUBMITTED; fell_back_to_mkt records
                # how it resolved. Keeping the limit price is deliberate — it is
                # the forensic record of what the order was trying to achieve.
                fell_back = 1
                fallback_reason, elapsed = "timeout", float(cfg.LIMIT_ORDER_TIMEOUT)
                self.events.append({
                    "event_id": _uid("evt"),
                    "occurred_at": (when + timedelta(seconds=cfg.LIMIT_ORDER_TIMEOUT))
                        .isoformat(sep=" ", timespec="seconds"),
                    "run_id": run_id, "event_type": "order_timeout", "severity": "warning",
                    "ticker": ticker, "tag": None, "order_id": order_id,
                    "detail": (f"LMT {order_id} on {ticker} unfilled after "
                               f"{cfg.LIMIT_ORDER_TIMEOUT}s; fell back to MKT."),
                    "remedy_action": "fallback_mkt", "resolved": 1, "resolution": "auto",
                })

        roll = rng.random()
        if roll < 0.05:
            status, filled, fill_status = "Partial", int(shares * rng.uniform(0.3, 0.8)), "partial"
        elif roll < 0.07:
            status, filled, fill_status = "Failed", 0, "unfilled"
        else:
            status, filled, fill_status = "Filled", shares, "complete"

        aggression = 1.0 if (order_type == "MKT" or fell_back) else 0.35
        slip = (spread / 2) * aggression * rng.uniform(0.8, 1.6)
        direction = 1 if side == "BUY" else -1
        fill_price = round(mid * (1 + direction * slip / 10_000), 2)

        self.orders.append({
            "order_id": order_id, "run_id": run_id, "ticker": ticker, "side": side,
            "order_type": order_type, "total_shares": shares, "limit_price": limit_price,
            "bid": bid, "ask": ask, "spread_bps": spread, "arrival_mid": mid,
            "batch_number": batch_number,
            "placed_at": when.isoformat(sep=" ", timespec="seconds"),
            "resolved_at": (when + timedelta(seconds=elapsed)).isoformat(sep=" ", timespec="seconds"),
            "elapsed_seconds": elapsed, "status": status, "filled_shares": filled,
            "fill_status": fill_status, "fallback_reason": fallback_reason,
            "fell_back_to_mkt": fell_back,
        })

        if filled:
            self.fills.append({
                "fill_id": _uid("fil"), "order_id": order_id, "quantity": filled,
                "price": fill_price,
                "filled_at": (when + timedelta(seconds=elapsed + 1)).isoformat(sep=" ", timespec="seconds"),
                "commission": round(filled * 0.005 + 0.35, 2),
            })

        # allocate fills back to contributing tags, pro rata
        for tag, pair, requested in contribs:
            alloc = int(requested * filled / shares) if shares else 0
            self.allocations.append({
                "allocation_id": _uid("alc"), "order_id": order_id, "tag": tag, "pair": pair,
                "requested_shares": requested, "allocated_shares": alloc,
                "allocation_status": ("complete" if alloc == requested
                                      else "partial" if alloc else "unfilled"),
            })

        if status == "Partial":
            self.events.append({
                "event_id": _uid("evt"),
                "occurred_at": (when + timedelta(seconds=elapsed)).isoformat(sep=" ", timespec="seconds"),
                "run_id": run_id, "event_type": "partial_fill", "severity": "warning",
                "ticker": ticker, "tag": None, "order_id": order_id,
                "detail": f"Requested {shares}, filled {filled}, unfilled {shares - filled}.",
                "remedy_action": "allocate_pro_rata", "resolved": 1, "resolution": "auto",
            })

    # ---- stages ----------------------------------------------------------
    def _stages(self, run_id, started, terminated, approved, executed, halted, day):
        rng = self.rng
        outputs = {
            1: {"position_count": len(self.open_positions), "staleness_minutes": round(rng.uniform(0.2, 4), 2)},
            2: {"duplicates_found": rng.randint(0, 2)},
            3: {"prices_fetched": len(self.open_positions) * 2, "failures": rng.randint(0, 1)},
            4: {"account_value": round(118_000 + rng.gauss(0, 3000), 2)},
            5: {"orphans_found": 0, "orphans_closed": 0},
            6: {"terminated_count": terminated},
            7: {"approved_count": len(approved), "rejected_count": rng.randint(8, 30)},
            8: {"executed_count": executed},
            9: {"stops_placed": executed, "stops_updated": len(self.open_positions)},
            10: {"mismatches_found": 1 if halted else 0},
            11: {"final_position_count": len(self.open_positions)},
        }
        cursor = started
        for number, name in cfg.WORKFLOW_STAGES:
            duration = rng.randint(120, 9000)
            status = "halted" if (halted and number == 10) else "ok"
            self.stages.append({
                "run_id": run_id, "stage_number": number, "stage_name": name,
                "status": status, "duration_ms": duration,
                "key_outputs": json.dumps(outputs.get(number, {})),
            })
            cursor += timedelta(milliseconds=duration)
            if status == "halted":
                break

    # ---- planted delisting ----------------------------------------------
    def _plant_delisting(self):
        closed = [p for p in self.positions if p["status"] == "closed"]
        if not closed:
            return
        pos = closed[len(closed) // 2]
        ticker = pos["co1"]
        pos["exit_reason"] = "Delisting"
        for inst in self.instruments:
            if inst["ticker"] == ticker:
                inst.update({"is_active": 0, "delisted_date": pos["termination_date"],
                             "delisting_type": "acquired", "acquirer": "PRIVATE_BUYER"})
                break
        self.events.append({
            "event_id": _uid("evt"),
            "occurred_at": f"{pos['termination_date']} 15:40:00",
            "run_id": None, "event_type": "delisting_detection", "severity": "warning",
            "ticker": ticker, "tag": pos["tag"], "order_id": None,
            "detail": (f"{ticker} detected as acquired. Position {pos['tag']} force-closed: "
                       f"acquirer shares liquidated, counterpart leg "
                       f"{pos['co2']} closed. Detection automated; execution operator-gated."),
            "remedy_action": "force_close", "resolved": 1, "resolution": "operator",
        })

    def _plant_wide_spread_fill(self):
        """Force one recent order to be a wide-spread market fill.

        Applied after generation rather than opportunistically inside the loop:
        the demo path and the eval set both depend on this case existing, and
        it should not depend on whether the RNG happened to place a suitable
        order in the final week.
        """
        recent = [o for o in self.orders
                  if o["status"] == "Filled" and o["side"] == "BUY"
                  and o["placed_at"][:10] >= self.calendar[-5].isoformat()]
        if not recent:
            recent = [o for o in self.orders if o["status"] == "Filled" and o["side"] == "BUY"]
        if not recent:
            return

        order = recent[len(recent) // 2]
        spread = round(self.rng.uniform(29.0, 36.0), 2)   # comfortably over the 24bps cap
        mid = order["arrival_mid"]
        half = mid * (spread / 10_000) / 2

        order.update({
            "spread_bps": spread,
            "bid": round(mid - half, 2),
            "ask": round(mid + half, 2),
            "order_type": "MKT",
            "limit_price": None,
            "fallback_reason": "spread_validation_failure",
            "fell_back_to_mkt": 0,
        })

        slip = (spread / 2) * self.rng.uniform(1.1, 1.4)
        for fill in self.fills:
            if fill["order_id"] == order["order_id"]:
                fill["price"] = round(mid * (1 + slip / 10_000), 2)
                break

        self.events.append({
            "event_id": _uid("evt"),
            "occurred_at": order["placed_at"],
            "run_id": order["run_id"], "event_type": "spread_validation_failure",
            "severity": "warning", "ticker": order["ticker"], "tag": None,
            "order_id": order["order_id"],
            "detail": (f"{order['ticker']} spread {spread}bps exceeded "
                       f"MAX_LIMIT_ORDER_SPREAD_BPS ({cfg.MAX_LIMIT_ORDER_SPREAD_BPS}); "
                       f"routed as MKT."),
            "remedy_action": "route_mkt", "resolved": 1, "resolution": "auto",
        })

    # ---- persistence -----------------------------------------------------
    def write(self, engine):
        payloads = [
            ("instruments", self.instruments), ("workflow_runs", self.runs),
            ("workflow_stages", self.stages), ("portfolio_snapshots", self.snapshots),
            ("pair_evaluations", self.evals), ("positions", self.positions),
            ("position_updates", self.updates), ("orders", self.orders),
            ("order_allocations", self.allocations), ("fills", self.fills),
            ("stop_orders", self.stops), ("risk_checks", self.risk_checks),
            ("system_events", self.events),
        ]
        with engine.begin() as conn:
            for table, rows in payloads:
                if not rows:
                    continue
                cols = ", ".join(rows[0].keys())
                binds = ", ".join(f":{c}" for c in rows[0].keys())
                for i in range(0, len(rows), 500):
                    conn.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({binds})"), rows[i:i + 500])


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a V9.4C-shaped synthetic blotter.")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db-url", default=None)
    args = ap.parse_args()

    engine = get_engine(args.db_url)
    create_schema(engine)
    gen = Generator(seed=args.seed, days=args.days)
    gen.run()
    gen.write(engine)

    print(f"Generated {args.days} trading days (seed={args.seed})\n")
    for table, count in table_counts(engine).items():
        print(f"  {table:<22} {count:>6}")


if __name__ == "__main__":
    main()
