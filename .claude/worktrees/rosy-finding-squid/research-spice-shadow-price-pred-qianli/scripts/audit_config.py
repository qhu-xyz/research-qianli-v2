"""Config drift audit — compares active champion config to current code defaults.

Usage:
    python scripts/audit_config.py [--registry-dir path/to/versions]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = PROJECT_ROOT / "versions"


def compare_dicts(a: dict, b: dict, path: str = "") -> list[tuple[str, str, str, str]]:
    """Recursively compare two dicts. Returns list of (path, status, a_val, b_val)."""
    diffs = []
    all_keys = sorted(set(list(a.keys()) + list(b.keys())))

    for key in all_keys:
        full_path = f"{path}.{key}" if path else key
        in_a = key in a
        in_b = key in b

        if in_a and not in_b:
            diffs.append((full_path, "ONLY IN STORED", str(a[key])[:60], "—"))
        elif in_b and not in_a:
            diffs.append((full_path, "ONLY IN CODE", "—", str(b[key])[:60]))
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            diffs.extend(compare_dicts(a[key], b[key], full_path))
        elif a[key] != b[key]:
            diffs.append((full_path, "DRIFT", str(a[key])[:60], str(b[key])[:60]))

    return diffs


def main():
    parser = argparse.ArgumentParser(description="Config drift audit")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir)
    manifest_path = registry_dir / "manifest.json"

    if not manifest_path.exists():
        print("No manifest.json found — no active champion to audit against.")
        sys.exit(0)

    with open(manifest_path) as f:
        manifest = json.load(f)

    active_id = manifest.get("active_version")
    if not active_id:
        print("No active version in manifest.")
        sys.exit(0)

    version_dir = registry_dir / active_id
    config_path = version_dir / "config.json"

    if not config_path.exists():
        print(f"Active version {active_id} has no config.json. Cannot audit.")
        sys.exit(1)

    with open(config_path) as f:
        stored_config = json.load(f)

    # Generate fresh config from current code
    from shadow_price_prediction.config import PredictionConfig
    from shadow_price_prediction.registry import ModelRegistry

    fresh_config = PredictionConfig()
    fresh_dict = ModelRegistry.config_to_dict(fresh_config)

    print(f"Config Audit: {active_id} (active champion)")
    print("=" * 60)
    print("Comparing stored config vs current code defaults...")
    print()

    diffs = compare_dicts(stored_config, fresh_dict)

    if not diffs:
        print("All fields: MATCH")
    else:
        matches = 0
        total_keys = len(set(list(stored_config.keys()) + list(fresh_dict.keys())))
        print(f"{'Field':<50} {'Status':<15} {'Stored':<30} {'Code Default':<30}")
        print("-" * 125)
        for path, status, a_val, b_val in diffs:
            print(f"{path:<50} {status:<15} {a_val:<30} {b_val:<30}")

    # Check source_commit drift
    print()
    meta_path = version_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(PROJECT_ROOT),
            )
            current_head = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            current_head = "unknown"

        champion_commit = meta.get("source_commit", "unknown")
        if champion_commit == current_head:
            print(f"Source commit: {champion_commit} (champion) == {current_head} (HEAD) — MATCH")
        else:
            print(f"Source commit: {champion_commit} (champion) vs {current_head} (HEAD) — WARN: code has diverged")

    # Check features.json consistency
    features_path = version_dir / "features.json"
    if features_path.exists() and config_path.exists():
        with open(features_path) as f:
            stored_features = json.load(f)
        s1_config = stored_config.get("features", {}).get("step1_features", [])
        s1_features_json = stored_features.get("step1_features", [])

        # Simple length comparison
        if len(s1_config) != len(s1_features_json):
            print(f"Feature ordering: MISMATCH (config has {len(s1_config)}, features.json has {len(s1_features_json)})")
        else:
            print(f"Feature ordering: MATCH ({len(s1_features_json)} step1 features)")

    # Overall
    print()
    if not diffs:
        print("Overall: PASS (no drift detected)")
    else:
        n_drifts = sum(1 for _, s, _, _ in diffs if s == "DRIFT")
        n_only = sum(1 for _, s, _, _ in diffs if s.startswith("ONLY"))
        print(f"Overall: {'FAIL' if n_drifts > 0 else 'WARN'} ({n_drifts} drifts, {n_only} asymmetric fields)")
        if n_drifts > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
