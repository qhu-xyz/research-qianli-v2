# Labeling Strategy: Core + Decayed Context Accumulation

## Overview
This document outlines the revised strategy for assigning shadow price labels to training data. The goal is to create a label that is robust to outage date uncertainty and extended outage durations, while maintaining the specificity of the planned outage window.

## Problem Statement
*   **Date Uncertainty:** Actual outages often shift by +/- 2 days from the planned start date.
*   **Variable Duration:** Outages may last longer than the standard 3-4 day planning assumption.
*   **Weather Patterns:** High-risk weather regimes (e.g., cold snaps) often span 5-10 days.
*   **Signal Loss:** A rigid 3-day sum can miss the actual binding event if it shifts slightly, training the model incorrectly (High Risk Features -> Zero Label).

## Proposed Solution: Core + Decayed Context
We employ a **Weighted Window Accumulation** strategy that prioritizes the planned dates ("Core") but also captures value from the surrounding period ("Context") with decaying weights.

### Algorithm
1.  **Define Windows**:
    *   **Core Window**: `outage_date` to `outage_date + 4 days`.
    *   **Context Window**: `outage_date - 7 days` to `outage_date + 11 days` (Total ~19 days, centered on Core).

2.  **Filter Noise**:
    *   Zero out any daily shadow price below a noise floor (default $1.0/MW).

3.  **Apply Linear Decay Weights**:
    *   **Core Days**: Weight = **1.0**.
    *   **Context Days**: Linear decay from 0.9 (adjacent to Core) to 0.1 (edges).
    *   Example Profile: `[0.1, 0.3, ..., 0.9, 1.0, 1.0, 1.0, 1.0, 0.9, ..., 0.1]`

4.  **Aggregate**:
    *   `Label = Sum(Daily_Shadow_Price * Daily_Weight)`

### Implementation Details

#### Weight Function
$$
W(d) =
\begin{cases}
1.0 & \text{if } d \in [T_{start}, T_{end}] \\
1.0 - \frac{|d - \text{Boundary}|}{7} \times 0.9 & \text{if } d \in \text{Context}
\end{cases}
$$

#### Pseudocode
```text
def calculate_label(da_label, outage_date):
    core_st = outage_date
    core_et = outage_date + 4 days
    context_st = core_st - 7 days
    context_et = core_et + 7 days

    # Get data slice
    window_data = da_label.loc[context_st:context_et]

    total_label = 0
    for day, price in window_data.items():
        if price < NOISE_FLOOR:
            continue

        if core_st <= day < core_et:
            weight = 1.0
        else:
            dist = min(abs((day - core_st).days), abs((day - (core_et - 1 day)).days))
            weight = max(0.1, 1.0 - (dist / 7.0))

        total_label += price * weight

    return total_label
```

## Benefits
*   **Robustness**: Captures value even if the outage shifts by a week.
*   **Precision**: Rewards accurate timing more than inaccurate timing (via decay).
*   **Duration Independent**: Automatically scales for longer outages.
