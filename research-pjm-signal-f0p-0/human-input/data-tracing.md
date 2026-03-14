1. "The rawest files are the DENSITY_SIGNAL_SCORE parquets right?"

  No. The rawest is DENSITY_DISTRIBUTION (77-bin probability distribution per constraint). The DENSITY_SIGNAL_SCORE is one level up — it's derived FROM
  the distribution (collapsing 77 bins into a single ML score).

  RAWEST                              MOST AGGREGATED
  ───────────────────────────────────────────────────────>

  DENSITY_DISTRIBUTION     DENSITY_SIGNAL_SCORE     V4.4 signal
    77 bins × 12,841          1 score × 12,841       20 percentiles × 1,227
    per outage_date            per outage_date        per (year, aq, ctype)
    (the raw simulation)       (ML summary)           (final curated signal)

  2. "Are prob_exceed_N features derived from the density files?"

  Yes, exactly. The 77 bins in the density distribution are a probability distribution of shadow price. To get prob_exceed_100, you sum all bins above
  $100. The V4.4 percentile features (100_max, 90_max, etc.) are this same idea but expressed as "P(flow exceeds X% of thermal limit)" — already
  aggregated across outage_dates and market_months by the signal production team.

  So V4.4 already contains the aggregated density distribution. We do NOT need to re-derive prob_exceed from raw density — it's already there as the 20
  percentile columns.

  3. "We need to aggregate/collapse into one row? Is this what V6.2B has done?"

  Yes. The raw data has 32 outage_dates × 3 market_months × 12,841 constraints = 821,824 rows per quarter. This must collapse to 1 row per (constraint,
  flow_direction) for training.

  The signal production team already did this aggregation:
  - V4.4: 32 × 3 outage_date slices → 1,227 rows (aggregated + filtered to curated universe)
  - V6.2B: same process for monthly, different formula → 489 rows

  So yes, V6.2B and V4.4 have already done the collapse for us. We don't need to re-aggregate from raw density.

  4. "Historical DA + the above are ALL sources of features?"

  Yes, three sources total:

  ┌───────────────────────────────────────────┬─────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │                  Source                   │                  What                   │                          Features                          │
  ├───────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ V4.4 signal (pre-aggregated)              │ Curated universe + collapsed            │ shadow_rank, shadow_price_da, deviation_max/sum_rank, 20   │
  │                                           │ predictions + historical DA             │ percentile features                                        │
  ├───────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Spice6 density_signal_score (needs        │ ML binding score — NOT in V4.4          │ score (mean across outage_dates/market_months)             │
  │ aggregation by us)                        │                                         │                                                            │
  ├───────────────────────────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Realized DA (computed by us)              │ Historical binding track record         │ bf_6/12/15, bfo_6/12 (binding frequency)                   │
  └───────────────────────────────────────────┴─────────────────────────────────────────┴────────────────────────────────────────────────────────────┘

  That's it. V4.4 already contains the aggregated density distribution (percentile features) AND the historical DA signal (shadow_rank). The only thing
  we add is:
  1. density_signal_score (independent ML prediction not in V4.4)
  2. binding_freq features (from our own realized DA computation)
  3. Ground truth labels (from realized DA)

  Bottom line: V4.4 did most of the heavy lifting. We enrich with one more spice6 feature (score) and our BF features, then re-rank with LightGBM.


## redesign the total universe
- pay attention to mapping

## compare universes

if change universes, need a way to compare quality. for example, each model in different universe's top20, do they capture more binding constraints or no? etcetc vc@20 might need to be interpreted differently (or just use the larger universe)
- need also examine different 