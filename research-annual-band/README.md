# Research Annual Band — Multi-RTO

## Structure

```
miso/           # MISO annual band research
pjm/            # PJM annual band research (not started)
```

## MISO Status
- V10 empirical asymmetric bands: dev + holdout evaluated
- Remaining gaps before production port:
  - E2E inference test not committed as reusable script
  - prod-port.md has stale monthly references in some sections
  - Production module (annual_band_generator.py) not built
- No shared annual-band interface exists yet

## PJM Status
- Not started

## Notes
- Scripts under miso/scripts/ have hardcoded paths referencing old layout in docstrings.
  ROOT is computed relative to script location so execution still works.
- No common/shared module exists. Interface will emerge from PJM implementation.
