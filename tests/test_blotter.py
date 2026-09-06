"""Invariant tests for the generated blotter.

These assert DOMAIN rules, not that code ran: disabled buckets never trade,
alpha stays in a market-neutral range, aggregated orders reconcile to their
allocations, and every planted demo scenario exists. If a change to the
generator breaks the economics, these fail rather than silently producing
plausible-looking nonsense.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src import config as cfg
from src.db import create_schema, get_engine
from src.generate_blotter import Generator


@pytest.fixture(scope="module")
def conn(blotter_url):
    engine = get_engine(blotter_url)
    create_schema(engine)
    gen = Generator(seed=42, days=90)
    gen.run()
    gen.write(engine)
    with engine.connect() as c:
        yield c


def scalar(conn, sql):
    return conn.execute(text(sql)).scalar()


# --- sizing rules ---------------------------------------------------------

def test_disabled_buckets_never_traded(conn):
    """Buckets with position_multiplier 0.0 must produce no positions."""
    disabled = [b for b, (_, _, m) in cfg.BUCKET_SIZING.items() if m == 0.0]
    placeholders = ", ".join(f"'{b}'" for b in disabled)
    count = scalar(conn, f"SELECT COUNT(*) FROM positions WHERE sum_dev_bucket IN ({placeholders})")
    assert count == 0


def test_position_multiplier_matches_bucket(conn):
    rows = conn.execute(text("SELECT DISTINCT sum_dev_bucket, position_multiplier FROM positions"))
    for bucket, multiplier in rows:
        assert multiplier == cfg.BUCKET_SIZING[bucket][2]


def test_leg_weights_sum_to_one(conn):
    bad = scalar(conn, "SELECT COUNT(*) FROM positions WHERE ABS(w1 + w2 - 1.0) > 1e-9")
    assert bad == 0


# --- alpha formula --------------------------------------------------------

def test_alpha_attaches_each_weight_to_its_own_leg():
    """w1 sizes Co1 and w2 sizes Co2. The weight that sized a leg must multiply
    that leg's return, whichever leg is long. External review found the U-tail
    calculation had them crossed."""
    from src.generate_blotter import Generator

    l_pos = {"tail": "L", "w1": 0.4, "w2": 0.6, "beta": 0.0}
    u_pos = {"tail": "U", "w1": 0.4, "w2": 0.6, "beta": 0.0}
    # Co1 flat, Co2 up 10%, index flat.
    assert Generator._alpha(l_pos, 0.0, 0.10, 0.0) == pytest.approx(-0.06)  # short Co2 at 0.6
    assert Generator._alpha(u_pos, 0.0, 0.10, 0.0) == pytest.approx(+0.06)  # long Co2 at 0.6
    # Co1 up 10%, Co2 flat.
    assert Generator._alpha(l_pos, 0.10, 0.0, 0.0) == pytest.approx(+0.04)  # long Co1 at 0.4
    assert Generator._alpha(u_pos, 0.10, 0.0, 0.0) == pytest.approx(-0.04)  # short Co1 at 0.4


def test_net_beta_sign_follows_tail():
    """For a U-tail the long leg is Co2, so net beta is w2*b2 - w1*b1 —
    the mirror of the L formula, not the same formula with legs relabelled."""
    from src.generate_blotter import Generator

    assert Generator._net_beta("L", 0.6, 0.4, 1.0, 1.0) == pytest.approx(+0.2)
    assert Generator._net_beta("U", 0.6, 0.4, 1.0, 1.0) == pytest.approx(-0.2)
    # Unequal betas: a lighter leg with higher beta can dominate.
    assert Generator._net_beta("L", 0.5, 0.5, 0.8, 1.2) == pytest.approx(-0.2)
    assert Generator._net_beta("U", 0.5, 0.5, 0.8, 1.2) == pytest.approx(+0.2)


# --- economics ------------------------------------------------------------

def test_alpha_is_market_neutral_scale(conn):
    """A hedged pair should not produce equity-like returns.

    Guards the correlated price paths: independent draws inflate alpha by an
    order of magnitude and the failure is invisible without this check.
    """
    lo = scalar(conn, "SELECT MIN(final_alpha_return_pct) FROM positions WHERE status='closed'")
    hi = scalar(conn, "SELECT MAX(final_alpha_return_pct) FROM positions WHERE status='closed'")
    assert -15.0 < lo and hi < 15.0


def test_all_exit_reasons_are_valid(conn):
    rows = conn.execute(text("SELECT DISTINCT exit_reason FROM positions WHERE exit_reason IS NOT NULL"))
    for (reason,) in rows:
        assert reason in cfg.EXIT_REASONS


def test_closed_positions_have_exit_fields(conn):
    incomplete = scalar(conn, """
        SELECT COUNT(*) FROM positions
        WHERE status='closed' AND (termination_date IS NULL
              OR final_alpha_return_pct IS NULL OR holding_days IS NULL)""")
    assert incomplete == 0


# --- execution ------------------------------------------------------------

def test_allocations_reconcile_to_orders(conn):
    """Aggregated order fills must fully distribute across contributing tags."""
    over = scalar(conn, """
        SELECT COUNT(*) FROM (
          SELECT o.order_id, o.filled_shares, SUM(a.allocated_shares) alloc
          FROM orders o JOIN order_allocations a ON a.order_id = o.order_id
          GROUP BY o.order_id) WHERE alloc > filled_shares""")
    assert over == 0


def test_limit_orders_respect_spread_cap(conn):
    """A resting LMT should never exist above MAX_LIMIT_ORDER_SPREAD_BPS."""
    violations = scalar(conn, f"""
        SELECT COUNT(*) FROM orders
        WHERE order_type='LMT' AND spread_bps > {cfg.MAX_LIMIT_ORDER_SPREAD_BPS}""")
    assert violations == 0


def test_timeouts_fell_back_to_market(conn):
    """A timed-out limit order stays typed LMT but is flagged as fallen back.

    order_type records what was submitted; fell_back_to_mkt records how it
    resolved. Both facts matter for execution analysis.
    """
    bad = scalar(conn, """
        SELECT COUNT(*) FROM orders
        WHERE fallback_reason='timeout' AND (fell_back_to_mkt=0 OR order_type<>'LMT')""")
    assert bad == 0


def test_market_orders_have_no_limit_price(conn):
    assert scalar(conn, "SELECT COUNT(*) FROM orders WHERE order_type='MKT' AND limit_price IS NOT NULL") == 0


def test_fallback_orders_retain_limit_price(conn):
    """The limit that failed to fill is forensic evidence — keep it."""
    missing = scalar(conn, """
        SELECT COUNT(*) FROM orders WHERE fell_back_to_mkt=1 AND limit_price IS NULL""")
    assert missing == 0


# --- risk and lifecycle ---------------------------------------------------

def test_every_position_has_a_stop(conn):
    orphaned = scalar(conn, """
        SELECT COUNT(*) FROM positions p
        LEFT JOIN stop_orders s ON s.tag = p.tag WHERE s.tag IS NULL""")
    assert orphaned == 0


def test_stop_tags_follow_convention(conn):
    bad = scalar(conn, "SELECT COUNT(*) FROM stop_orders WHERE stop_order_tag NOT LIKE 'SQLOSS_%'")
    assert bad == 0


def test_triggered_stops_produced_orphan_events(conn):
    triggered = scalar(conn, "SELECT COUNT(*) FROM stop_orders WHERE status='triggered'")
    orphans = scalar(conn, "SELECT COUNT(*) FROM system_events WHERE event_type='orphan_detection'")
    assert triggered > 0 and orphans == triggered


# --- planted scenarios ----------------------------------------------------

@pytest.mark.parametrize("event_type", [
    "reconciliation_mismatch", "delisting_detection",
    "order_timeout", "partial_fill", "spread_validation_failure",
])
def test_planted_scenario_present(conn, event_type):
    """The demo and eval set depend on these existing every run."""
    count = scalar(conn, f"SELECT COUNT(*) FROM system_events WHERE event_type='{event_type}'")
    assert count > 0


def test_wide_spread_slippage_case_exists(conn):
    """There must be at least one order with a clear, explainable bad fill."""
    worst = scalar(conn, """
        SELECT MAX(ABS(f.price - o.arrival_mid) / o.arrival_mid * 10000)
        FROM orders o JOIN fills f ON f.order_id = o.order_id""")
    assert worst > 15.0


def test_delisted_instrument_is_marked_inactive(conn):
    assert scalar(conn, "SELECT COUNT(*) FROM instruments WHERE is_active=0 AND delisted_date IS NOT NULL") >= 1


# --- workflow -------------------------------------------------------------

def test_every_run_has_stages(conn):
    runs_without = scalar(conn, """
        SELECT COUNT(*) FROM workflow_runs r
        LEFT JOIN workflow_stages s ON s.run_id = r.run_id
        WHERE s.run_id IS NULL""")
    assert runs_without == 0


def test_halted_runs_stop_early(conn):
    """A halted run should not log all eleven stages."""
    rows = conn.execute(text("""
        SELECT r.run_id, COUNT(s.stage_number) FROM workflow_runs r
        JOIN workflow_stages s ON s.run_id = r.run_id
        WHERE r.outcome='halted' GROUP BY r.run_id"""))
    for _, stage_count in rows:
        assert stage_count < len(cfg.WORKFLOW_STAGES)


def test_generation_is_deterministic(tmp_path):
    """Same seed, same blotter — the demo and evals depend on it."""
    counts = []
    for i in range(2):
        engine = get_engine(f"sqlite:///{tmp_path / f'det{i}.db'}")
        create_schema(engine)
        gen = Generator(seed=7, days=30)
        gen.run()
        gen.write(engine)
        with engine.connect() as c:
            counts.append((
                c.execute(text("SELECT COUNT(*) FROM positions")).scalar(),
                c.execute(text("SELECT COUNT(*) FROM orders")).scalar(),
                c.execute(text("SELECT ROUND(SUM(final_alpha_return_pct), 6) FROM positions")).scalar(),
            ))
    assert counts[0] == counts[1]
