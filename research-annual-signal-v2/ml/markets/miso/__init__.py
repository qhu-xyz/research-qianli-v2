"""ml.markets.miso — MISO-specific annual signal implementation.

Current production code lives in ml/ (flat layout). This package is the
migration target. During migration, compatibility shims in ml/ will
delegate to modules here.

Ownership (future):
  - config.py: MISO-specific constants, paths, round calendar
  - bridge.py: MISO CID-to-branch mapping (SPICE bridge + supplement)
  - data_loader.py: MISO density/limits loading
  - realized_da.py: MISO DA shadow price cache (monthly + daily)
  - ground_truth.py: MISO branch-level GT construction
  - history_features.py: MISO BF/da_rank/NB features
  - signal_publisher.py: MISO annual signal publication

Current status: skeleton only. No code moved yet.
"""
