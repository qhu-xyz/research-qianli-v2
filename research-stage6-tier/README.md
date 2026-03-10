# Stage 6 Tier

This stage is dedicated to fixing the partial-month leakage problem without
dropping all recent information.

Current design target:

- keep the safe row boundary from `v10e-lag1` (training rows stop at `M-2`)
- rebuild historical realized-DA features during dataset construction
- allow the most recent prior month `M-1` to contribute only through
  `look_back_days`
- verify that `look_back_days=31` reproduces full-month historical features
  when the aggregation semantics are preserved

The first implementation focuses on `f0` and the `binding_freq_*` feature
family because that is where the confirmed leakage and the largest lift live.

Important caveats:

- `look_back_days=31` is only an exact-equivalence check for the rebuilt
  `binding_freq_*` family in stage6. It does not prove the entire stage6
  dataset matches the old stage5 monthly collapse.
- Exact equivalence requires preserving signed daily netting and reconstructing
  monthly realized SP as `abs(sum(daily_net))`. Caching only daily absolute
  values would be wrong.
- The cutoff is currently interpreted as an inclusive calendar day inside the
  prior month. If the production run happens before the end of that day, the
  cutoff should be shifted earlier at the timestamp level instead of using the
  whole day.
