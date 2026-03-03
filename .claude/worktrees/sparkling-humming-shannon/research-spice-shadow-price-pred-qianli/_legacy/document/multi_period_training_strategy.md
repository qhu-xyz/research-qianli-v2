# Multi-Period Training Data Selection Strategy

## 1. Context & Objective
*   **Goal**: Train a unified model to predict shadow prices for any given `(Auction Month, Market Month)` pair.
*   **Data Structure**: Density data is already stored by `(Auction Month, Market Month)`. Constraint data is stored by `(Auction Month, Period Type)`.
*   **Strategy**: Pool historical data from the previous 12 months, expanding all available period types into their constituent monthly samples.

## 2. MISO FTR Auction Schedule
**Planning Year**: June - May.

| Auction Month | Available Period Types |
| :--- | :--- |
| **6 (Jun)** | `['f0']` |
| **7 (Jul)** | `['f0', 'f1', 'q2', 'q3', 'q4']` |
| **8 (Aug)** | `['f0', 'f1', 'f2', 'f3']` |
| **9 (Sep)** | `['f0', 'f1', 'f2']` |
| **10 (Oct)** | `['f0', 'f1', 'q3', 'q4']` |
| **11 (Nov)** | `['f0', 'f1', 'f2', 'f3']` |
| **12 (Dec)** | `['f0', 'f1', 'f2']` |
| **1 (Jan)** | `['f0', 'f1', 'q4']` |
| **2 (Feb)** | `['f0', 'f1', 'f2', 'f3']` |
| **3 (Mar)** | `['f0', 'f1', 'f2']` |
| **4 (Apr)** | `['f0', 'f1']` |
| **5 (May)** | `['f0']` |

## 3. Period Type to Market Month Mapping
To load data, we must map each `Period Type` to its corresponding `Market Month(s)`.

*   **f-series** (Monthly):
    *   `f0`: $M = A$
    *   `f1`: $M = A + 1$
    *   `f2`: $M = A + 2$
    *   `f3`: $M = A + 3$
*   **q-series** (Seasonal Strips):
    *   `q2`: **Sep, Oct, Nov**
    *   `q3`: **Dec, Jan, Feb**
    *   `q4`: **Mar, Apr, May**

## 4. Training Data Construction

### 4.1. Lookback Logic
For a target Auction Month $A_{target}$:
1.  **Window**: $A_{target} - 12$ months to $A_{target} - 1$ month.
2.  **Iterate**: For each historical Auction Month $A_{hist}$ in the window:
    a.  **Get Periods**: Lookup available `period_types` from the Schedule Table.
    b.  **Expand**: For each `period_type`, identify the list of `Market Months` ($M_{hist}$).
    c.  **Load Pair**: For each resulting `(A_{hist}, M_{hist})` pair:
        *   **Density**: Load from `.../auction_month={A}/market_month={M}/...`
        *   **Constraints**: Load from `.../auction_month={A}/.../period_type={P}/...`
        *   **Horizon**: Calculate $H = (M_{hist} - A_{hist})$ in months.

### 4.2. Example: Training for Oct 2025
*   **Historical Auction: July 2025** ($A=7$)
    *   **f0** -> Load `(Jul, Jul)`. Constraint Period: `f0`. Horizon: 0.
    *   **f1** -> Load `(Jul, Aug)`. Constraint Period: `f1`. Horizon: 1.
    *   **q2** ->
        *   Load `(Jul, Sep)`. Constraint Period: `q2`. Horizon: 2.
        *   Load `(Jul, Oct)`. Constraint Period: `q2`. Horizon: 3.
        *   Load `(Jul, Nov)`. Constraint Period: `q2`. Horizon: 4.
    *   ... and so on for `q3`, `q4`.

## 5. Feature Engineering
The model will be trained on this pooled dataset containing various horizons.
*   **`forecast_horizon`**: Explicit feature (0, 1, 2, ...) to indicate uncertainty level.
*   **`seasonality`**: Cyclic encoding of the Market Month to capture grid physics.

## 6. Implementation Steps

1.  **Config Update**:
    *   Add `AUCTION_SCHEDULE` constant.
    *   Add `PERIOD_MAPPING` logic (e.g., `q2` -> `[9, 10, 11]`).

2.  **DataLoader Update**:
    *   Modify `load_training_data` to implement the expansion logic described in 4.1.
    *   Ensure `load_data_for_outage` accepts a `constraint_period_type` argument (distinct from the implied period type of the density file) to ensure the correct constraint info is loaded.

3.  **Training**:
    *   Train a single model on the combined dataset.
    *   Validate that `forecast_horizon` is being used effectively.
