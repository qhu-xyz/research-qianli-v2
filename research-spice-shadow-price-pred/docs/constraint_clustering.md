# Constraint Clustering & Branch Grouping Strategy

## Problem
New constraints often appear due to grid upgrades, topology changes (re-configurations), or naming convention updates. These "New" constraints have zero training history, yet they often represent the same physical bottleneck (or a very nearby one) as an "Old" constraint that has ample history. Treating them as separate entities loses valuable signal.

## Recommended Approaches

### 1. Semantic Grouping (Name Similarity)
**Best for**: Handling naming changes or minor re-configurations (e.g., "Line 123" vs "Line 123 Section 2").

*   **Logic**: Use string similarity (Levenshtein distance, Jaccard similarity) on the `monitored_facility` name.
*   **Algorithm**:
    1.  Tokenize constraint names (remove "FLO", "TO", "KV", numbers).
    2.  For every *New* constraint, find the *Old* constraint with the highest similarity score.
    3.  If Score > Threshold (e.g., 0.8), map New -> Old (or map both to a common "Canonical Name").
*   **Example**: `WEST_BUS_1_LN` and `WEST_BUS_1_LN_REBUILD` -> Group: `WEST_BUS_1`.

### 2. Topological Grouping (Substation/Bus Matching)
**Best for**: "Next-door neighbor" constraints where a limitation shifted slightly (e.g., from a line to its transformer).

*   **Logic**: Extract the "End Buses" or Substations from the constraint definition.
*   **Algorithm**:
    1.  Parse constraint string limits (e.g., `345 LC_CREEK-345 P_VIEW`) to extract locations `{LC_CREEK, P_VIEW}`.
    2.  Build a graph where nodes are Substations and edges are constraints.
    3.  **Cluster Constraints** that share at least one common node (or are 1 hop away) AND share the same voltage class.
*   **Application**: If `Line A-B` binds often, and suddenly `Transformer A` binds, the model for `Transformer A` should inherit the history of `Line A-B`.

### 3. Flowgate / Interface Aggregation
**Best for**: Regional transfers where multiple parallel lines share the load.

*   **Logic**: Define "Super-Constraints" (Flowgates) that aggregate multiple physical elements.
*   **Algorithm**:
    1.  Identify predefined Interfaces (e.g., "AP South", "Central East").
    2.  Map *any* constituent line of that interface to the Interface ID.
    3.  **Training**: Train the model on the **Interface Shadow Price** (Max or Sum of constituents).
    4.  **Prediction**: Predict for the Interface. If a specific line binds, it's just a manifestation of the Interface binding.

### 4. Implementation Recommendation: "Proxy Model" Fallback

Instead of complex graph theory, implement a **Hierarchical Fallback**:

1.  **Exact Match**: Use specific branch model if history > N samples.
2.  **Proxy Match (The Feature)**: Add a `proxy_branch_name` column to your mapping table.
    - Manually or algorithmically populate this for known upgrades (e.g., `New_Line_A` maps to `Old_Line_A`).
    - Train on the `proxy_branch_name`'s history.
3.  **Cluster Default**: If no proxy, fall back to the **Driver-Region-Voltage Group** model (Method defined in `region_strategy.md`).

This ensures that a new, empty constraint `New_Line` falls back to `Region_Default`, which is much better than a random guess, while `Relabeled_Line` falls back to its true history.
