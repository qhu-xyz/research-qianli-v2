# Review: `task_plan.md` (Second Pass)

## Findings

1. **Phase 3a still uses invalid MISO baseline benchmarks.**

   In [task_plan.md:253](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:253), the plan labels `~792/~532/~457` as old MAEs. Those values still do not line up with the archived corrected-unit benchmarks:

   - R1 nodal-f0 corrected MAE is `264` average monthly, with quarter MAEs `307/293/257/201` in [miso/archive_v1/docs/v10-consolidation-report.md:56](/home/xyz/workspace/research-qianli-v2/research-annual-band/miso/archive_v1/docs/v10-consolidation-report.md:56).
   - `532` and `457` are R2/R3 quarterly P95 half-widths, not MAEs, in [miso/archive_v1/docs/v10-consolidation-report.md:173](/home/xyz/workspace/research-qianli-v2/research-annual-band/miso/archive_v1/docs/v10-consolidation-report.md:173).

   This is still the main integrity blocker. Phase 3a will validate against the wrong target metric unless the benchmark table is corrected.

2. **The Phase 1c consistency check is still too weak to catch the exact unit bug already documented in the archive.**

   In [task_plan.md:142](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:142), the MISO consistency gate is still “old `mcp_mean` vs new `mcp` or equivalent, corr > 0.999”. That would pass a pure scale error such as `new_mcp = 3 * old_mcp_mean`.

   The archive already documents that the prior failure mode was a monthly/quarterly mismatch in [miso/archive_v1/docs/v10-consolidation-report.md:7](/home/xyz/workspace/research-qianli-v2/research-annual-band/miso/archive_v1/docs/v10-consolidation-report.md:7), and the corrected convention is explicit in [miso/archive_v1/docs/v10-consolidation-report.md:50](/home/xyz/workspace/research-qianli-v2/research-annual-band/miso/archive_v1/docs/v10-consolidation-report.md:50).

   The verification gate needs exact column mapping plus scale checks, not correlation alone. At minimum:

   - verify whether new `mcp` is quarterly or monthly,
   - verify whether new `mcp_mean` exists and satisfies `mcp_mean = mcp / 3` for MISO,
   - reject matches that only hold up to a constant factor.

3. **The leakage rules for `recent_3mo` and `mar_rev` are still asserted as safe before availability is actually verified.**

   In [task_plan.md:232](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:232) and [task_plan.md:233](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:233), `recent_3mo` and `mar_rev` are marked “tight but safe”.

   But the source-of-truth request is stricter in:

   - [human-input.md:22](/home/xyz/workspace/research-qianli-v2/research-annual-band/human-input.md:22)
   - [human-input.md:39](/home/xyz/workspace/research-qianli-v2/research-annual-band/human-input.md:39)

   The user explicitly asked for this to be checked against the real early-April auction cutoff. March-based features may be fine, but the plan should treat them as conditional on verified settlement/data-availability timing, not as pre-cleared safe inputs.

4. **Phase 7 is closer to auditable now, but it still does not pin the exact runtime and production touchpoints the port will modify.**

   The plan now correctly references `pmodel` at a high level in [task_plan.md:432](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:432), and the target code does exist under:

   - [pmodel/src/pmodel/base/ftr24/v1/band_generator.py:1659](/home/xyz/workspace/pmodel/src/pmodel/base/ftr24/v1/band_generator.py:1659)
   - [pmodel/src/pmodel/base/ftr24/v1/autotuning.py:387](/home/xyz/workspace/pmodel/src/pmodel/base/ftr24/v1/autotuning.py:387)

   But the task plan still does not say explicitly that annual integration spans both files, and it still says “load old data from archive paths” in [task_plan.md:137](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:137) even though the actual archived scripts point to `/opt/temp/qianli/annual_research/...`:

   - [miso/archive_v1/scripts/run_v9_bands.py:48](/home/xyz/workspace/research-qianli-v2/research-annual-band/miso/archive_v1/scripts/run_v9_bands.py:48)
   - [pjm/archive_v1/scripts/run_v1_bands.py:59](/home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/archive_v1/scripts/run_v1_bands.py:59)

   This matters because the current workspace still cannot import `pbase`, so the environment bootstrap is not implicit. A truly auditable plan should pin:

   - the exact `pmodel` file paths to inspect and modify,
   - the Python environment / interpreter,
   - `pbase` availability,
   - Ray initialization requirements,
   - concrete old/new parquet locations.

## Open Questions

1. **Is “P95 coverage” in Phase 3b meant to reproduce archived two-sided coverage, or is it meant to be a one-sided buy-clearing metric?**

   The reporting standard says buy clearing is primary in [task_plan.md:55](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:55), but Phase 3b’s comparison table in [task_plan.md:278](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:278) only says “P95 coverage”. The archive values `93.2/89.6/92.0` came from two-sided coverage, so the plan should name that explicitly if reproducibility is the goal.

2. **Are the Phase 1c pass thresholds intentionally loose for a data-integrity gate?**

   In [task_plan.md:141](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:141) and [task_plan.md:145](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:145), the plan allows 5% row drift and only 95% path overlap. On a multi-million-row dataset, that can hide a very large mismatch. That may still be acceptable as an initial triage threshold, but it does not read like a final integrity gate.

## Verification Notes

- Two first-pass issues are now fixed in `task_plan.md`: the plan has an explicit path-level dedup step in [task_plan.md:119](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:119), and the Phase 5 cell accounting is now internally consistent in [task_plan.md:376](/home/xyz/workspace/research-qianli-v2/research-annual-band/task_plan.md:376).
- The top-level archive summary in `task_plan.md` is still broadly consistent with the saved MISO/PJM reports.
- The old parquet inputs referenced by the archived scripts still exist on disk under `/opt/temp/qianli/annual_research/`.
- The “35 archived Python scripts” claim is correct.
