"""ml.products.annual — annual product contracts.

Owns:
  - Universe definitions (from docs/contracts/universe-catalog.md)
  - Feature recipe definitions (from docs/contracts/feature-recipes.md)
  - Label recipe definitions
  - Published output schema contracts
  - Cross-model comparison runners

Does NOT own:
  - RTO-specific loaders (those belong in ml/markets/{rto}/)
  - Model training logic (those belong in ml/markets/{rto}/ or scripts/)
"""
