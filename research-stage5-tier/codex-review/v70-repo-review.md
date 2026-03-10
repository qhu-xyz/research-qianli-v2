# Review: `research-miso-signal7` Current State

Reviewed repo:
- [research-miso-signal7](/home/xyz/workspace/research-qianli-v2/research-miso-signal7)

## Findings

1. **High: The repo does not contain the automated validation entrypoint it claims to use.**

[v70-validation-plan.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-validation-plan.md#L10) says:

```text
Run scripts/validate_v70.py after generating signals. All checks are automated.
```

But there is no `scripts/validate_v70.py` anywhere in the repo, and I do not see saved validation outputs either. That means the claim “all validation gates passed” is not reproducible from the current checkout.

2. **High: Cache preflight is incomplete for correctness on a fresh or partial cache.**

[cache.py](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/cache.py#L33) only fetches realized DA months required for training labels:

- collect usable training months
- map each to `delivery_month(train_month, ptype)`
- fetch those months only

It does **not** fetch the older realized months needed for `binding_freq_1/3/6/12/15`.

But [inference.py](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/inference.py#L66) builds binding frequency from whatever months are present in the cache:

```python
prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
```

So in a fresh or partially populated environment, generation can succeed with silently shortened BF lookbacks. That is not deployment-safe.

3. **Medium: The fallback loader is too permissive and can hide real V7.0 failures.**

[loader.py](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/loader.py#L27) catches any exception, then silently falls back to V6.2B:

```python
try:
    df = ConstraintsSignal(...V7...).load_data(...)
    if len(df) > 0:
        return df, signal_name
except Exception:
    pass

df = ConstraintsSignal(...V6.2B...).load_data(...)
return df, fallback_signal
```

This means corrupted V7.0 output, schema bugs, transient read failures, or other unexpected errors can masquerade as a normal fallback. Returning `source_signal_name` helps only if every caller actively checks it.

4. **Medium: The stated coverage/fallback story is inaccurate relative to what is actually on disk.**

The agent summary said:

- `f0`: 27 months (`2024-01 -> 2026-03`)
- `f1`: 23 months
- fallback to V6.2B before `2024-01`

That is not the current state of the written signal.

I verified that V7.0 exists on disk much earlier:

- `f0`: **75 months**, `2020-01 -> 2026-03`
- `f1`: **63 months**, `2020-01 -> 2026-03` (with expected schedule gaps)

I also spot-checked that older ML months differ from V6.2B in exactly the ML columns:

- `2020-06 / f0 / onpeak`
- `2021-01 / f0 / onpeak`
- `2023-05 / f0 / onpeak`

all exist under V7.0 and differ from V6.2B in `rank_ori`, `rank`, and `tier`.

So the fallback-before-2024-01 story is no longer true for the current signal contents.

## What Looks Good

- The core generation path is real and implemented:
  - [generate_v70_signal.py](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/generate_v70_signal.py)
  - [inference.py](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/inference.py)
  - [signal_writer.py](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/signal_writer.py)
- ML slices preserve schema, row count, and index order in spot checks.
- The current tiering design is explicit and documented in [v70-design-choices.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-design-choices.md).
- Passthrough slices appear exact in spot checks, for example `q4/onpeak 2024-01`.

## Bottom Line

I would not call `research-miso-signal7` “completely done.”

The generation code exists and the signal has been written, but:

- the validation claim is not reproducible from the repo,
- cache preflight is not robust enough for a fresh environment,
- and the fallback/coverage description is out of sync with the actual written signal.

The highest-priority fix is:

1. add a real `validate_v70.py` plus saved validation output,
2. fix cache preflight to include BF lookback requirements,
3. tighten the fallback loader so unexpected V7.0 read failures are surfaced rather than silently downgraded.
