"""V9.4C configuration constants, mirrored from the live system.

These values appear in log records and drive filter outcomes, so the generator
must use the same numbers the real system does — otherwise generated data
implies thresholds that were never applied.
"""

MAX_ACCOUNT_LEVERAGE = 1.9
EMERGENCY_LEVERAGE_THRESHOLD = 1.8
PREFILTER_MAX_SPREAD_BPS = 38.0
MAX_LIMIT_ORDER_SPREAD_BPS = 24.0
LIMIT_ORDER_TIMEOUT = 45           # seconds
STOP_LOSS_ALPHA_THRESHOLD = 0.40   # 40% alpha deterioration
MAX_INDEX_GROSS_EXPOSURE_PCT = 40.0
MAX_NEW_POSITIONS_PER_TICKER = 1
MIN_POSITION_SIZE = 100.0
MAX_POSITION_SIZE = 5000.0

# Bucket -> (W1, W2, position_multiplier) for the Lower tail.
# Upper tail mirrors with reversed weights. Multiplier 0.0 disables the bucket.
BUCKET_SIZING = {
    "0-10%":   (0.60, 0.40, 1.3),
    "10-20%":  (0.55, 0.45, 1.1),
    "20-30%":  (0.50, 0.50, 1.0),
    "30-40%":  (0.50, 0.50, 0.7),
    "40-50%":  (0.50, 0.50, 0.0),
    "50-60%":  (0.50, 0.50, 0.0),
    "60-70%":  (0.50, 0.50, 0.0),
    "70-80%":  (0.50, 0.50, 0.7),
    "80-90%":  (0.55, 0.45, 1.1),
    "90-100%": (0.60, 0.40, 1.4),
}

BUCKETS = list(BUCKET_SIZING)

WORKFLOW_STAGES = [
    (1,  "workflow.portfolio_load"),
    (2,  "workflow.duplicate_removal"),
    (3,  "workflow.price_update"),
    (4,  "workflow.metrics_calculation"),
    (5,  "workflow.orphan_detection"),
    (6,  "workflow.termination_evaluation"),
    (7,  "workflow.trade_evaluation"),
    (8,  "workflow.trade_execution"),
    (9,  "workflow.stop_placement"),
    (10, "workflow.reconciliation"),
    (11, "workflow.portfolio_save"),
]

EXIT_REASONS = [
    "Date Reached", "Earnings Alert", "Alpha Reached",
    "Stop Loss Triggered", "Delisting", "Manual",
]

PRIMARY_FAIL_REASONS = [
    "spread_hurdle", "earnings_filter", "trend_filter",
    "same_direction", "nominal_direction", "tstat",
]

REJECTION_REASONS = [
    "bucket_not_tradeable", "duplicate", "leverage", "position_size",
    "max_per_ticker", "index_concentration", "factor_exposure", "factor_shock",
]

FACTORS = ["momentum", "value", "size", "quality", "low_vol"]
