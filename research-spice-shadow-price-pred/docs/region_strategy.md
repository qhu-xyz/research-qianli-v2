# Driver-Based Region Strategy

## Problem Statement
Default transmission zones (e.g., AEP in PJM, "North" in MISO) are often too geographically broad to serve as effective grouping features for shadow price prediction. They group together branches with fundamentally different congestion drivers (e.g., Wind vs. Gas vs. Load).

## Proposed Solution
Define **"Driver-Based Regions"** that group branches based on their primary economic and physical drivers rather than just administrative boundaries.

## Region Definitions

### 1. PJM Interconnection

| Driver Region | Constituents | Primary Drivers | Key Features |
| :--- | :--- | :--- | :--- |
| **PJM_West_Wind** | `ComEd`, `AEP` (West/Dayton), `Dayton` | **Wind**, Nuclear Base Load | Negative covariance with wind; often decoupled from gas. |
| **PJM_East_Gas** | `PSEG`, `PECO`, `BGE`, `PEPCO`, `Delmarva`, `AECO` | **Natural Gas**, Import Limits | High correlation with Transco Z6/Tetco M3 gas; load-driven peaks. |
| **PJM_South_Transfer** | `Dominion`, `APS` | **Transfers**, Load Growth | North-to-South flow constraints; Data center load impact. |
| **PJM_Peninsula** | `Penelec`, `MetEd` | **Interface Flows** | Constraints between West Gen and East Load (AP South interface). |

### 2. MISO

| Driver Region | Constituents | Primary Drivers | Key Features |
| :--- | :--- | :--- | :--- |
| **MISO_West_Wind** | Zones 1, 3 (`MN`, `IA`, `ND`, `SD`) | **Extreme Wind**, Stability | Phantom congestion; often $0 flows due to stability limits. |
| **MISO_Central** | Zones 2, 4, 6, 7 (`WI`, `MI`, `IL`, `IN`) | **Thermal/Load** | Traditional supply/demand; correlated with coal/gas spark spreads. |
| **MISO_South** | Zones 8, 9, 10 (`AR`, `LA`, `MS`, `TX`) | **Gas**, N-S Transfer | Distinct separation from North; driven by Southern gas markets and heat. |

## Implementation Plan

### Step 1: Mapping
Create a helper function `get_driver_region(original_zone: str, iso: str) -> str` in `data_loader.py`.

### Step 2: Feature Engineering
- Add `driver_region` as a categorical feature in the dataset.
- Use this feature for **Cluster-Based Default Models** (e.g., train one default model for `Wind_Driven` branches and another for `Gas_Driven`).

### Step 3: Analysis
- Analyze prediction error residuals by these groups to verify if noise is reduced.

## Voltage Grouping Strategy

### Rationale
Voltage level is a proxy for the *role* a line plays in the grid.
- **High Voltage**: Bulk transfer, wind export, prone to regional weather/stability limits.
- **Lower Voltage**: Load serving, prone to local outages and demand spikes.

### Definitions

| Voltage Group | Range (kV) | Description | Key Congestion Drivers |
| :--- | :--- | :--- | :--- |
| **Bulk / EHV** | `≥ 345 kV` | Backbone Grid (765, 500, 345 kV) | Regional transfers, Stability limits, Wind export patterns. |
| **Regional / HV** | `230 kV` | Intra-zonal Backbone | Bridge between bulk grid and load centers. often constraint limited by step-down transformers. |
| **Local / Sub-T** | `≤ 161 kV` | Sub-transmission (138, 115, 69 kV) | Local load peaks, N-1 maintenance outages, specific generator trips. |

### Implementation
1.  **Extract Voltage**: Parse from `constraint_name` or `monitored_facility` string (matches regex `\d{3}`).
2.  **Binning**:
    *   `Bulk`: `>= 300`
    *   `Regional`: `>= 200` & `< 300`
    *   `Local`: `< 200`

### Transformers (XFMR) Special Handling
Transformers connect two voltage levels (e.g., 345kV / 138kV).
- **Rule**: Classify by the **Low-Side Voltage**.
- **Rationale**: A step-down transformer typically acts as the "gateway" or bottleneck feeding the lower voltage network. Its congestion patterns (load peaks, local constraints) are far more correlated with the lower voltage group than the high-voltage bulk grid.
    - *Example*: A 345/138 kV transformer should be grouped with **138 kV (Local/Sub-T)**.
    - *Exception*: EHV Autos (e.g., 765/500 kV) where the low side is still Bulk (>= 345 kV) remain in the **Bulk** group.
