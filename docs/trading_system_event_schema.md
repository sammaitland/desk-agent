# V9.4C Trade Event Schema — Mock Log Reference

Use this file to generate realistic historical trade logs for the desk agent's Postgres blotter.

---

## Event Categories at a Glance

| # | Category | Count | Description |
|---|----------|-------|-------------|
| 1 | Signal & Screening | 13 | Universe filtering, scoring, factor checks |
| 2 | Order & Execution | 9 | Evaluation, aggregation, fills |
| 3 | Position Lifecycle | 8 | Open, scale, exit (by reason) |
| 4 | Risk Checks | 7 | Leverage, beta, concentration, exposure |
| 5 | Portfolio-Level | 7 | Load, append, save, reconcile |
| 6 | P&L & Accounting | 6 | Alpha, leg returns, completion records |
| 7 | Data Pipeline | 7 | Fetches, caches, calendar updates |
| 8 | Stop Loss | 8 | Calculate, place, update, orphan detect |
| 9 | Errors & Reconciliation | 10 | Connection, partial fills, mismatches |

**Total: ~75 distinct event types**

---

## 1. Signal & Screening Events

### 1.1 Primary Filter Application
- **Source**: `Helper/Tool_Box.py` → `apply_primary_filters_with_leniency_v92()`
- **Fields**: `pair, index, tstat, spread_bps, earnings_days_out, filter_result (Pass/Fail), fail_reason`
- **Triggers**: Every pair in the universe, every run

### 1.2 Secondary Signal Calculation
- **Source**: `Implementation/LAM.py` → `apply_secondary_signals()`
- **Fields**: `pair, volume_ratio, rolling_intraday_vol, IV_percentile, volume_dominance, true_last_hour_volatility`
- **Triggers**: All pairs that pass primary filters

### 1.3 Composite Score Calculation
- **Source**: `Implementation/LAM.py` → `calculate_composite_score()`
- **Fields**: `pair, weighted_score, percentile_bands_used, stability_weights_used, composite_score`
- **Triggers**: All pairs with secondary signals

### 1.4 Alpha Sum Deviation Calculation
- **Source**: `Implementation/Pre_Filter.py` → `calculate_sum_deviation_from_historical_v92()`
- **Fields**: `pair, sum_deviation_15d, sum_dev_percentile`
- **Triggers**: All pairs passing filters

### 1.5 CDF Bucket Assignment
- **Source**: `Helper/Tool_Box.py` → `assign_sum_dev_bucket()`
- **Fields**: `pair, sum_dev_percentile, bucket (e.g. "0-10%", "90-100%")`
- **Bucket values**: `0-10%, 10-20%, 20-30%, 30-40%, 40-50%, 50-60%, 60-70%, 70-80%, 80-90%, 90-100%`

### 1.6 Trending Stock Filter
- **Source**: `Helper/Tool_Box.py` → `check_trend_filter()`
- **Fields**: `pair, co1_trending (bool), co2_trending (bool), result (Pass/Fail)`

### 1.7 Earnings Filter
- **Source**: `Helper/Tool_Box.py` → `check_earnings_filter()`
- **Fields**: `pair, co1_days_to_earnings, co2_days_to_earnings, result (Pass/Fail)`

### 1.8 Spread Hurdle Check
- **Source**: `Helper/Tool_Box.py` → `check_spread_hurdle()`
- **Fields**: `pair, weighted_spread_bps, max_spread_bps (38 prefilter / 24 limit order), result (Pass/Fail)`

### 1.9 Same Direction Check
- **Source**: `Helper/Tool_Box.py` → `check_same_direction()`
- **Fields**: `pair, co1_direction, co2_direction, result (Pass/Fail)`

### 1.10 Nominal Direction (5-Day Momentum) Check
- **Source**: `Helper/Tool_Box.py` → `check_nominal_direction()`
- **Fields**: `pair, co1_5d_return, co2_5d_return, result (Pass/Fail)`

### 1.11 Factor Shock Detection
- **Source**: `Helper/Factor_Shock_Detection.py` → `get_live_factor_status()`
- **Fields**: `factor_name, z_score, threshold, is_shocked (bool)`

### 1.12 At-Risk Pair Identification
- **Source**: `Helper/Factor_Shock_Detection.py` → `get_pairs_at_risk()`
- **Fields**: `pair, shocked_factor, net_exposure, action (SUPPRESS/ALLOW)`

### 1.13 Pair Suppression
- **Source**: `Implementation/Pre_Filter.py`
- **Fields**: `pair, suppression_reason, shocked_factors[]`

---

## 2. Order & Execution Events

### 2.1 Trade Evaluation
- **Source**: `Helper/Portfolio_Management.py` → `evaluate_trades()`
- **Fields**: `pair, index, tail, composite_score, spread_quality_score, sum_dev_extremity_score, composite_priority_score, evaluation_result (Approved/Rejected), rejection_reason`

### 2.2 Duplicate Check
- **Source**: `Helper/Portfolio_Management.py` → `check_for_existing_trades()`
- **Fields**: `pair, already_in_portfolio (bool), existing_tag`

### 2.3 Tradeable Bucket Filter
- **Source**: `Helper/Constraints.py` → `is_tradeable_bucket()`
- **Fields**: `pair, bucket, position_multiplier, is_tradeable (bool)`
- **Note**: Buckets with position_multiplier = 0.0 are not tradeable (40-50%, 50-60%, 60-70%)

### 2.4 Composite Priority Scoring
- **Source**: `Helper/Portfolio_Management.py` → `calculate_composite_priority_score()`
- **Fields**: `pair, weighted_score, spread_quality, sum_dev_extremity, composite_priority_score`

### 2.5 Leverage Constraint Check (Pre-Trade)
- **Source**: `Helper/Trade_Execution.py` → `check_leverage_before_trade()`
- **Fields**: `current_leverage, proposed_trade_notional, post_trade_leverage, max_leverage (1.9), result (Pass/Fail)`

### 2.6 Order Aggregation
- **Source**: `Helper/Trade_Execution.py` → `OrderAggregator.aggregate_approved_trades()`
- **Fields**: `ticker, direction (BUY/SELL), total_shares, contributing_pairs[], contributing_tags[]`
- **Note**: Groups orders by ticker+direction to reduce commissions

### 2.7 Order Allocation (Post-Fill)
- **Source**: `Helper/Trade_Execution.py` → `OrderAggregator.allocate_fills()`
- **Fields**: `ticker, filled_shares, allocation_per_pair[], fill_status (complete/partial/unfilled)`

### 2.8 Batch Execution
- **Source**: `Helper/Trade_Execution.py` → `execute_trades_in_batches()`
- **Fields**: `batch_number, pairs_in_batch, batch_start_time, batch_end_time`

### 2.9 Single Pair Execution
- **Source**: `Helper/Trade_Execution.py` → `execute_single_pair_trade()`
- **Fields**: `tag, pair, co1, co2, tail, quantity1, quantity2, co1_fill_price, co2_fill_price, co1_order_type (LIMIT/MKT), co2_order_type (LIMIT/MKT), entry_spread_bps, status (Filled/Partial/Failed/Skipped), execution_timestamp`

### Limit Order Sub-Events
- **Place**: `place_limit_order()` → `ticker, side, quantity, limit_price, order_id`
- **Monitor**: `monitor_limit_order_fill()` → `order_id, elapsed_seconds, fill_status, filled_qty, timeout (45s)`
- **Fallback**: Market order if limit doesn't fill → `order_id, fallback_reason (timeout)`
- **Spread Validation**: `validate_spread_for_limit_order()` → `ticker, bid, ask, spread_bps, max_spread_bps (24), result`

---

## 3. Position Lifecycle Events

### 3.1 Position Open
- **Fields (Portfolio row)**:
  ```
  Tag                     — unique trade identifier (e.g. "VGT_AAPL_MSFT_L_20260815_001")
  Pair                    — e.g. "AAPL_MSFT"
  Co1, Co2                — component tickers
  Index                   — sector ETF (VGT, VHT, VFH, VIS, VCR, VOX)
  Tail                    — L (long Co1/short Co2) or U (short Co1/long Co2)
  Quantity1, Quantity2    — share counts per leg
  W1, W2                  — leg weights (e.g. 0.60/0.40)
  Trade Value Co1/Co2 ($) — dollar notional per leg
  Total_Notional          — sum of both legs
  Position_Multiplier     — bucket-based sizing (0.7x–1.4x)
  Co1/Co2 at Initiation   — entry prices
  Index at Initiation     — index price at entry
  Trade Initiation Date   — entry timestamp
  Sum_Dev_Bucket          — CDF bucket
  Sum_Deviation           — 15-day sum alpha deviation
  Sum_Dev_Percentile      — CDF percentile
  Version                 — V9 or V9.2
  Stop_Price              — initial stop price
  Stop_Order_ID           — IBKR order ID for stop
  ```

### 3.2 Position Sizing by Bucket
| Tail | Bucket | W1 | W2 | Position Multiplier |
|------|--------|----|----|---------------------|
| Lower | 0-10% | 0.60 | 0.40 | 1.3x |
| Lower | 10-20% | 0.55 | 0.45 | 1.1x |
| Lower | 20-30% | 0.50 | 0.50 | 1.0x |
| Lower | 30-40% | 0.50 | 0.50 | 0.7x |
| Lower | 40-50% | 0.50 | 0.50 | 0.0x (disabled) |
| Lower | 50-60% | 0.50 | 0.50 | 0.0x (disabled) |
| Lower | 60-70% | 0.50 | 0.50 | 0.0x (disabled) |
| Lower | 70-80% | 0.50 | 0.50 | 0.7x |
| Lower | 80-90% | 0.55 | 0.45 | 1.1x |
| Lower | 90-100% | 0.60 | 0.40 | 1.4x |
| Upper | (mirror structure with reversed weights) | | | |

### 3.3 Live Position Update (Daily)
- **Fields**: `tag, pair, live_co1_price, live_co2_price, live_index_price, live_alpha_return_pct, days_held`

### 3.4 Position Exit — by Reason

| Exit Reason | Trigger | Source |
|-------------|---------|--------|
| `Date Reached` | Holding period expired | `evaluate_trade_terminations()` |
| `Earnings Alert` | Earnings announcement imminent | `evaluate_trade_terminations()` |
| `Alpha Reached` | Alpha >= early exit hurdle | `check_early_exit()` |
| `Stop Loss Triggered` | Short leg stop hit | Stop loss module |
| `Delisting` | M&A / ticker delisted | `Delisting_Handler` |
| `Manual` | Force-terminated via config | `FORCE_TERMINATE_TAGS` |

### 3.5 Completed Trade Record
- **Fields (appended to Completed_Trades.xlsx)**:
  ```
  (all Position Open fields, plus:)
  Exit_Reason
  Trade Termination Date
  Holding_Days
  Co1/Co2 at Exit
  Index at Exit
  Co1_Return_Pct, Co2_Return_Pct
  Index_Return_Pct
  Co1_Alpha_Pct, Co2_Alpha_Pct
  Final_Alpha_Return_Pct
  Entry_Spread_BPS
  Weighted_Score
  Composite_Score
  ```

---

## 4. Risk Check Events

### 4.1 Portfolio Beta Check
- **Fields**: `current_beta, min_beta, max_beta, result (Pass/Fail)`

### 4.2 Leverage Limit Check
- **Fields**: `current_leverage, max_leverage (1.9x), result (Pass/Fail)`
- **Source**: `Constraints.check_leverage_limit()`

### 4.3 Emergency Leverage Check
- **Fields**: `current_leverage, emergency_threshold (1.8x), trading_halted (bool)`
- **Source**: `Constraints.check_emergency_leverage()`

### 4.4 Position Size Validation
- **Fields**: `pair, proposed_size, min_size ($100), max_size ($5000), result (Pass/Fail)`

### 4.5 Max Positions Per Ticker
- **Fields**: `ticker, current_new_positions, max_per_ticker (1), result (Pass/Fail)`

### 4.6 Index Concentration Check
- **Fields**: `index, index_gross_pct, max_pct (40%), result (Pass/Fail)`
- **Source**: `Constraints.check_index_concentration()`

### 4.7 Factor Exposure Check
- **Fields**: `factor, current_net_exposure, limit, result (Pass/Fail)`
- **Source**: Config `FACTOR_EXPOSURE_LIMITS`

---

## 5. Portfolio-Level Events

| Event | Fields |
|-------|--------|
| **Portfolio Load** | `timestamp, position_count, staleness_minutes` |
| **Duplicate Tag Removal** | `duplicates_found, duplicates_removed, backup_path` |
| **Trade Append** | `new_trades_count, post_append_position_count` |
| **Portfolio Save** | `timestamp, position_count, file_path` |
| **Daily Closes Capture** | `date, tickers_captured, tickers_failed` |
| **Portfolio Beta Calculation** | `dollar_weighted_beta, position_count` |
| **Gross Exposure Calculation** | `total_gross_exposure, account_value, leverage` |

---

## 6. P&L & Accounting Events

### 6.1 Alpha Calculation (per position, per update)
```
alpha = W1 * co1_return - W2 * co2_return - beta * index_return   (L-tail)
alpha = W1 * co2_return - W2 * co1_return - beta * index_return   (U-tail, reversed)
```
- **Fields**: `tag, pair, tail, co1_return_pct, co2_return_pct, index_return_pct, beta, live_alpha_return_pct`

### 6.2 Leg Returns
- **Fields**: `tag, co1_init, co1_current, co1_return_pct, co2_init, co2_current, co2_return_pct`

### 6.3 Component Alpha
- **Fields**: `tag, co1_alpha_pct (co1_ret - beta*index_ret), co2_alpha_pct (co2_ret - beta*index_ret)`

### 6.4 Trade Completion
- **Destination**: `Completed_Trades.xlsx`
- **Fields**: See section 3.5

### 6.5 Daily Terminated Trades
- **Destination**: `daily_terminated_trades.xlsx`
- **Fields**: Same as completed trade record, batched by day

### 6.6 Execution Summary
- **Destination**: `Execution_Summary.xlsx`
- **Fields**: `Status, Tag, Pair, Index, Tail, Entry_Spread_BPS, Composite_Score, Execution_Timestamp`

---

## 7. Data Pipeline Events

| Event | Fields |
|-------|--------|
| **Historical Data Fetch** | `tickers[], lookback_days (365), source, cache_hit (bool), cache_path` |
| **Live Prices Fetch** | `tickers[], source (IBKR), timestamp, success_count, fail_count` |
| **DGS10 Yield Fetch** | `date, yield_value, source (FRED)` |
| **Alpha Cache Operation** | `pair, cache_hit (bool), cache_key` |
| **Analyst Data Archive** | `ticker, source (Alpha Vantage), db_path (analyst_data.db)` |
| **Earnings Calendar Update** | `index, tickers_updated, source (IBKR Wall Street Horizons)` |
| **Options/IV Cache** | `ticker, iv_percentile, cache_path` |

---

## 8. Stop Loss Events

| Event | Fields |
|-------|--------|
| **Stop Price Calculation** | `tag, short_leg_ticker, entry_price, stop_price, alpha_threshold (0.40)` |
| **Short Leg Identification** | `tag, short_leg (co1 or co2), ticker, entry_price, quantity` |
| **Stop Order Placement** | `tag, stop_order_tag (SQLOSS_{tag}), ticker, stop_price, quantity, order_id` |
| **Existing Stop Retrieval** | `active_stops_count, stop_order_ids[]` |
| **Stop Order Cancellation** | `order_id, reason` |
| **Stop Price Update** | `tag, old_stop_price, new_stop_price, trigger (nightly recalc)` |
| **Orphan Detection** | `tag, orphaned_leg (long), short_leg_stopped (bool)` |
| **Orphan Closure** | `tag, ticker, quantity, side (SELL), order_type (MKT), fill_price` |

---

## 9. Error & Reconciliation Events

| Event | Fields |
|-------|--------|
| **IBKR Connection Error** | `timestamp, error_message, retry_count` |
| **Missing Market Data** | `ticker, data_type, fallback_used (bool)` |
| **Spread Validation Failure** | `ticker, bid, ask, spread_bps, max_allowed_bps` |
| **Leverage Exceeded** | `current_leverage, threshold, action (halt_trading)` |
| **Index Concentration Breach** | `index, current_pct, max_pct, action (reject_trade)` |
| **Factor Shock Exposure** | `pair, factor, exposure, action (suppress)` |
| **Partial Fill** | `order_id, ticker, requested_qty, filled_qty, unfilled_qty` |
| **Order Timeout** | `order_id, ticker, elapsed_seconds, timeout_threshold (45s), fallback (MKT)` |
| **Delisting Detection** | `ticker, event_type, acquirer, detection_date` |
| **Reconciliation Mismatch** | `ticker, tws_position, portfolio_position, discrepancy, remedy_action` |

---

## 10. Workflow Stage Events (Execution Timeline)

Each daily run logs these stages in order:

| Stage | Event Name | Key Outputs |
|-------|------------|-------------|
| 1 | `workflow.portfolio_load` | position_count, staleness |
| 2 | `workflow.duplicate_removal` | duplicates_found |
| 3 | `workflow.price_update` | prices_fetched, failures |
| 4 | `workflow.metrics_calculation` | account_value, leverage, beta |
| 5 | `workflow.orphan_detection` | orphans_found, orphans_closed |
| 6 | `workflow.termination_evaluation` | terminated_count, by_reason{} |
| 7 | `workflow.trade_evaluation` | shortlist_count, approved_count, rejected_count |
| 8 | `workflow.trade_execution` | executed_count, filled_count, failed_count |
| 9 | `workflow.stop_placement` | stops_placed, stops_updated |
| 10 | `workflow.reconciliation` | mismatches_found, remedies_applied |
| 11 | `workflow.portfolio_save` | final_position_count |
| Post | `workflow.post_close_update` | closes_captured, daily_data_saved |

---

## Key Enums / Status Values

```
Order Status:      Filled | Partial | Failed | Skipped | Executed
Fill Allocation:   complete | partial | unfilled
Filter Result:     Pass | Fail
Pair Action:       ALLOW | SUPPRESS
Tail:              L (Lower) | U (Upper)
Exit Reason:       Date Reached | Earnings Alert | Alpha Reached |
                   Stop Loss Triggered | Delisting | Manual
Index:             VGT | VHT | VFH | VIS | VCR | VOX
Order Type:        LMT | MKT
Side:              BUY | SELL
Version:           V9 | V9.2
```

---

## Key Config Parameters to Embed in Logs

| Parameter | Value | Use |
|-----------|-------|-----|
| BASE_TRADE_SIZE | (config) | Notional per trade |
| MAX_ACCOUNT_LEVERAGE | 1.9x | Hard cap |
| EMERGENCY_LEVERAGE_THRESHOLD | 1.8x | Halt trigger |
| PREFILTER_MAX_SPREAD_BPS | 38 | Primary filter |
| MAX_LIMIT_ORDER_SPREAD_BPS | 24 | Execution filter |
| LIMIT_ORDER_TIMEOUT | 45s | Fallback trigger |
| STOP_LOSS_ALPHA_THRESHOLD | 0.40 | 40% deterioration |
| MAX_INDEX_GROSS_EXPOSURE_PCT | 40% | Concentration cap |
| MAX_NEW_POSITIONS_PER_TICKER | 1 | Per-run limit |
| MIN_POSITION_SIZE | $100 | Floor |
| MAX_POSITION_SIZE | $5000 | Ceiling |

---

## Output Files (Where Logs Lived)

| File | Path Pattern | Content |
|------|-------------|---------|
| Portfolio | `Master Implementation/Portfolio.xlsx` | Active positions |
| Completed Trades | `Master Implementation/Completed_Trades.xlsx` | Closed positions with P&L |
| Execution Summary | `Master Implementation/Execution_Summary.xlsx` | Per-run execution results |
| Daily Terminated | `Master Implementation/daily_terminated_trades.xlsx` | Day's exits |
| Rejected Archive | `Master Implementation/Rejected_Trades_Archive.xlsx` | Filtered-out pairs with reasons |
| Shortlist | `V9.3/V9_Shortlist.xlsx` | Primary filter output |
| Longlist | `V9.3/V9_Longlist.xlsx` | Secondary signal output |
| Log File | `~/Desktop/v9c_trading.log` | Rotating text log (10MB x 5) |
| Daily Closes | `Master Implementation/daily_closes.xlsx` | Historical close prices |
