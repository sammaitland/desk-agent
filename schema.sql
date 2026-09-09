-- Desk Agent — blotter schema (V9.4C-aligned)
--
-- Built from the V9.4C Trade Event Schema. Field names, enums and thresholds
-- mirror the live system so that real paper-account output populates these
-- tables unchanged.
--
-- Portable SQL: runs on SQLite (dev) and PostgreSQL (target).
--
-- OPEN QUESTIONS flagged during design (resolve before this is load-bearing):
--   1. Longlist/Shortlist ordering contradicts ARCHITECTURE.md. This schema
--      follows the event-schema reference: primary filters -> Shortlist,
--      secondary signals -> Longlist.
--   2. Index enum here includes VOX; README names five ETFs. Six assumed.

DROP TABLE IF EXISTS order_allocations;
DROP TABLE IF EXISTS fills;
DROP TABLE IF EXISTS stop_orders;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS position_updates;
DROP TABLE IF EXISTS positions;
DROP TABLE IF EXISTS pair_evaluations;
DROP TABLE IF EXISTS risk_checks;
DROP TABLE IF EXISTS portfolio_snapshots;
DROP TABLE IF EXISTS workflow_stages;
DROP TABLE IF EXISTS workflow_runs;
DROP TABLE IF EXISTS system_events;
DROP TABLE IF EXISTS instruments;


CREATE TABLE instruments (
    ticker          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    idx             TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    delisted_date   TEXT,
    delisting_type  TEXT,
    acquirer        TEXT
);


CREATE TABLE workflow_runs (
    run_id          TEXT PRIMARY KEY,
    run_date        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    outcome         TEXT NOT NULL
);


CREATE TABLE workflow_stages (
    run_id          TEXT NOT NULL,
    stage_number    INTEGER NOT NULL,
    stage_name      TEXT NOT NULL,
    status          TEXT NOT NULL,
    duration_ms     INTEGER,
    key_outputs     TEXT,
    PRIMARY KEY (run_id, stage_number),
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
);


CREATE TABLE portfolio_snapshots (
    run_id                  TEXT PRIMARY KEY,
    snapshot_date           TEXT NOT NULL,
    position_count          INTEGER NOT NULL,
    -- Account value and leverage come from the broker at run time, not from
    -- the portfolio file. Nullable so a snapshot can be built from the file
    -- alone and enriched from the log when the parser exists.
    account_value           REAL,
    total_gross_exposure    REAL NOT NULL,
    leverage                REAL,
    dollar_weighted_beta    REAL,
    staleness_minutes       REAL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
);


CREATE TABLE pair_evaluations (
    eval_id                     TEXT PRIMARY KEY,
    run_id                      TEXT NOT NULL,
    evaluated_at                TEXT NOT NULL,
    pair                        TEXT NOT NULL,
    idx                         TEXT NOT NULL,
    co1                         TEXT NOT NULL,
    co2                         TEXT NOT NULL,
    tail                        TEXT,
    tstat                       REAL,
    weighted_spread_bps         REAL,
    earnings_days_out           INTEGER,
    co1_trending                INTEGER,
    co2_trending                INTEGER,
    same_direction_result       TEXT,
    nominal_direction_result    TEXT,
    -- Nullable: only the longlist reports a Pass/Fail primary result. The
    -- prefilter reports Active 0/1 (-> evaluation_result) and the rejected
    -- archive reports neither. The synthetic generator always supplies one.
    primary_result              TEXT,
    primary_fail_reason         TEXT,
    volume_ratio                REAL,
    rolling_intraday_vol        REAL,
    iv_percentile               REAL,
    volume_dominance            REAL,
    true_last_hour_volatility   REAL,
    weighted_score              REAL,
    composite_score             REAL,
    sum_deviation_15d           REAL,
    sum_dev_percentile          REAL,
    sum_dev_bucket              TEXT,
    position_multiplier         REAL,
    is_tradeable_bucket         INTEGER,
    shocked_factors             TEXT,
    factor_action               TEXT,
    spread_quality_score        REAL,
    sum_dev_extremity_score     REAL,
    composite_priority_score    REAL,
    evaluation_result           TEXT,
    rejection_reason            TEXT,
    -- Added when the adapter met the real V9.3 files. The live system's
    -- Reason is free text embedding ticker and magnitude ("Trending filter:
    -- ASAN - Negative trending: -73.49% excess return over 12M"), so the
    -- category and the original are kept apart. `stage` records how far a
    -- pair got: prefilter -> longlist -> shortlist, or rejected on score.
    -- `source_tag` is the parameters-file row index, which is what the
    -- pre-trade files key on — position tags do not exist until execution.
    rejection_detail            TEXT,
    stage                       TEXT,
    version                     TEXT,
    source_tag                  INTEGER,
    category1                   TEXT,
    category2                   TEXT,
    index_bias                  REAL,
    alpha_cdf                   REAL,
    earnings_result             TEXT,
    spread_result               TEXT,
    alpha_result                TEXT,
    two_day_result              TEXT,
    trend_result                TEXT,
    score_threshold             REAL,
    score_shortfall             REAL,
    co1_price                   REAL,
    co2_price                   REAL,
    index_price                 REAL,
    co1_beta                    REAL,
    co2_beta                    REAL,
    w1                          REAL,
    w2                          REAL,
    volume_ratio_pct            REAL,
    intraday_vol_pct            REAL,
    volume_dominance_pct        REAL,
    last_hour_pct               REAL,
    iv_percentile_pct           REAL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
);


-- Most columns are nullable because the real system's files leave them so.
-- The synthetic generator always supplied every field, which made the
-- constraints look safe; the first real Completed_Trades.xlsx had nulls in
-- ten NOT NULL columns — Index at Exit missing on 78 of 185 rows, Sum_Dev
-- fields on 13, and the whole row blank on one. Only what identifies a
-- position is required.
CREATE TABLE positions (
    tag                     TEXT PRIMARY KEY,
    pair                    TEXT NOT NULL,
    co1                     TEXT NOT NULL,
    co2                     TEXT NOT NULL,
    idx                     TEXT NOT NULL,
    tail                    TEXT NOT NULL,
    version                 TEXT,
    quantity1               INTEGER,
    quantity2               INTEGER,
    w1                      REAL,
    w2                      REAL,
    trade_value_co1         REAL,
    trade_value_co2         REAL,
    total_notional          REAL,
    position_multiplier     REAL,
    co1_at_initiation       REAL,
    co2_at_initiation       REAL,
    index_at_initiation     REAL,
    trade_initiation_date   TEXT NOT NULL,
    entry_spread_bps        REAL,
    sum_dev_bucket          TEXT,
    sum_deviation           REAL,
    sum_dev_percentile      REAL,
    weighted_score          REAL,
    composite_score         REAL,
    beta                    REAL,
    stop_price              REAL,
    stop_order_id           TEXT,
    status                  TEXT NOT NULL,
    exit_reason             TEXT,
    -- The live system's Exit_Reason is free text carrying the detail:
    -- "Early Exit - Day15_TakeProfit_8pct", "Earnings - NSSC reports
    -- 2026-02-02", "Pre-Holiday Exit (term date 2025-12-25 is non-trading
    -- day)". The canonical reason above is queryable; this keeps the day
    -- number, the reporting ticker or the displaced date.
    exit_detail             TEXT,
    scheduled_termination_date TEXT,
    index_bias              REAL,
    -- Secondary-signal snapshot at entry, carried by both real position files.
    volume_ratio            REAL,
    rolling_intraday_vol    REAL,
    volume_dominance        REAL,
    iv_percentile           REAL,
    termination_date        TEXT,
    holding_days            INTEGER,
    co1_at_exit             REAL,
    co2_at_exit             REAL,
    index_at_exit           REAL,
    co1_return_pct          REAL,
    co2_return_pct          REAL,
    index_return_pct        REAL,
    co1_alpha_pct           REAL,
    co2_alpha_pct           REAL,
    final_alpha_return_pct  REAL,
    FOREIGN KEY (co1) REFERENCES instruments(ticker),
    FOREIGN KEY (co2) REFERENCES instruments(ticker)
);


CREATE TABLE position_updates (
    tag                     TEXT NOT NULL,
    update_date             TEXT NOT NULL,
    run_id                  TEXT,
    -- Leg prices and returns are nullable: the synthetic generator always
    -- supplies them, but the live system's Portfolio.xlsx carries the alpha
    -- figure without the per-leg marks. The alpha is the required field.
    live_co1_price          REAL,
    live_co2_price          REAL,
    live_index_price        REAL,
    co1_return_pct          REAL,
    co2_return_pct          REAL,
    index_return_pct        REAL,
    co1_alpha_pct           REAL,
    co2_alpha_pct           REAL,
    live_alpha_return_pct   REAL NOT NULL,
    days_held               INTEGER NOT NULL,
    PRIMARY KEY (tag, update_date),
    FOREIGN KEY (tag) REFERENCES positions(tag)
);


CREATE TABLE orders (
    order_id            TEXT PRIMARY KEY,
    run_id              TEXT,
    ticker              TEXT NOT NULL,
    side                TEXT NOT NULL,
    order_type          TEXT NOT NULL,
    total_shares        INTEGER NOT NULL,
    limit_price         REAL,
    bid                 REAL,
    ask                 REAL,
    spread_bps          REAL,
    arrival_mid         REAL,
    batch_number        INTEGER,
    placed_at           TEXT NOT NULL,
    resolved_at         TEXT,
    elapsed_seconds     REAL,
    status              TEXT NOT NULL,
    filled_shares       INTEGER NOT NULL DEFAULT 0,
    fill_status         TEXT,
    fallback_reason     TEXT,
    fell_back_to_mkt    INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (ticker) REFERENCES instruments(ticker),
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
);


CREATE TABLE order_allocations (
    allocation_id       TEXT PRIMARY KEY,
    order_id            TEXT NOT NULL,
    tag                 TEXT NOT NULL,
    pair                TEXT NOT NULL,
    requested_shares    INTEGER NOT NULL,
    allocated_shares    INTEGER NOT NULL,
    allocation_status   TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);


CREATE TABLE fills (
    fill_id     TEXT PRIMARY KEY,
    order_id    TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    price       REAL NOT NULL,
    filled_at   TEXT NOT NULL,
    commission  REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);


CREATE TABLE stop_orders (
    stop_order_tag  TEXT PRIMARY KEY,
    tag             TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    stop_price      REAL NOT NULL,
    alpha_threshold REAL NOT NULL DEFAULT 0.40,
    order_id        TEXT,
    placed_at       TEXT NOT NULL,
    last_updated_at TEXT,
    status          TEXT NOT NULL,
    triggered_at    TEXT,
    FOREIGN KEY (tag) REFERENCES positions(tag)
);


CREATE TABLE risk_checks (
    check_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    checked_at      TEXT NOT NULL,
    check_name      TEXT NOT NULL,
    subject         TEXT,
    current_value   REAL,
    threshold       REAL,
    result          TEXT NOT NULL,
    action          TEXT,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
);


CREATE TABLE system_events (
    event_id        TEXT PRIMARY KEY,
    occurred_at     TEXT NOT NULL,
    run_id          TEXT,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    ticker          TEXT,
    tag             TEXT,
    order_id        TEXT,
    detail          TEXT NOT NULL,
    remedy_action   TEXT,
    resolved        INTEGER NOT NULL DEFAULT 0,
    resolution      TEXT
);


CREATE INDEX idx_orders_placed      ON orders(placed_at);
CREATE INDEX idx_orders_ticker      ON orders(ticker);
CREATE INDEX idx_orders_run         ON orders(run_id);
CREATE INDEX idx_fills_order        ON fills(order_id);
CREATE INDEX idx_alloc_order        ON order_allocations(order_id);
CREATE INDEX idx_alloc_tag          ON order_allocations(tag);
CREATE INDEX idx_eval_run           ON pair_evaluations(run_id);
CREATE INDEX idx_eval_pair          ON pair_evaluations(pair);
CREATE INDEX idx_eval_result        ON pair_evaluations(evaluation_result);
CREATE INDEX idx_positions_status   ON positions(status);
CREATE INDEX idx_positions_idx      ON positions(idx);
CREATE INDEX idx_updates_date       ON position_updates(update_date);
CREATE INDEX idx_events_occurred    ON system_events(occurred_at);
CREATE INDEX idx_events_type        ON system_events(event_type);
CREATE INDEX idx_risk_run           ON risk_checks(run_id);
CREATE INDEX idx_stages_run         ON workflow_stages(run_id);
