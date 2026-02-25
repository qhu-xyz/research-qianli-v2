"""Registry integrity audit — checks all versions for required files and checksums.

Usage:
    python scripts/audit_registry.py [--registry-dir path/to/versions]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = PROJECT_ROOT / "versions"


def verify_git_commit(commit: str) -> bool:
    """Check if a commit hash is valid in this repo."""
    if commit in ("unknown", "", None):
        return False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", commit],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode == 0
    except Exception:
        return False


def audit_version(version_dir: Path) -> dict:
    """Audit a single version directory. Returns dict with status and details."""
    from shadow_price_prediction.naming import REQUIRED_VERSION_FILES
    from shadow_price_prediction.registry import _compute_checksum

    result = {
        "version": version_dir.name,
        "status": "PASS",
        "issues": [],
        "files_present": 0,
        "files_required": len(REQUIRED_VERSION_FILES),
    }

    # Check required files
    for fname in REQUIRED_VERSION_FILES:
        if (version_dir / fname).exists():
            result["files_present"] += 1
        else:
            result["issues"].append(f"MISSING: {fname}")

    # Load meta.json
    meta_path = version_dir / "meta.json"
    if not meta_path.exists():
        result["status"] = "FAIL"
        result["issues"].append("meta.json not found — cannot verify integrity")
        return result

    with open(meta_path) as f:
        meta = json.load(f)

    # Verify config checksum
    config_path = version_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        actual_checksum = _compute_checksum(config_data)
        expected_checksum = meta.get("config_checksum")
        if expected_checksum and actual_checksum != expected_checksum:
            result["issues"].append(
                f"CONFIG CHECKSUM MISMATCH: expected {expected_checksum}, got {actual_checksum}"
            )

    # Verify version hash
    version_hash = meta.get("version_hash")
    if version_hash:
        import hashlib
        artifact_names = [
            "config.json", "features.json", "metrics.json",
            "threshold_manifest.json", "feature_importance.json", "train_manifest.json",
        ]
        hasher = hashlib.sha256()
        for name in artifact_names:
            fpath = version_dir / name
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                canonical = json.dumps(data, sort_keys=True, default=str)
                hasher.update(canonical.encode("utf-8"))
            else:
                hasher.update(b"__MISSING__")
        recomputed = "sha256:" + hasher.hexdigest()[:32]
        if recomputed != version_hash:
            result["issues"].append(
                f"VERSION HASH MISMATCH: stored {version_hash}, recomputed {recomputed}"
            )

    # Verify source_commit
    commit = meta.get("source_commit", "unknown")
    if commit != "unknown":
        if not verify_git_commit(commit):
            result["issues"].append(f"SOURCE COMMIT NOT FOUND: {commit}")

    # Determine final status
    critical_issues = [i for i in result["issues"] if "MISMATCH" in i or "NOT FOUND" in i]
    missing_count = len(REQUIRED_VERSION_FILES) - result["files_present"]

    if critical_issues:
        result["status"] = "FAIL"
    elif missing_count > 0:
        result["status"] = "WARN"

    return result


def audit_manifest(registry_dir: Path) -> dict:
    """Audit the top-level manifest.json."""
    manifest_path = registry_dir / "manifest.json"
    result = {"status": "PASS", "issues": [], "active_version": None}

    if not manifest_path.exists():
        result["status"] = "WARN"
        result["issues"].append("manifest.json not found (no active champion)")
        return result

    with open(manifest_path) as f:
        manifest = json.load(f)

    active = manifest.get("active_version")
    result["active_version"] = active

    if active is None:
        result["status"] = "WARN"
        result["issues"].append("No active version set in manifest")
        return result

    # Check active version exists
    active_dir = registry_dir / active
    if not active_dir.exists():
        result["status"] = "FAIL"
        result["issues"].append(f"Active version directory not found: {active}")
        return result

    # Check active version hash matches
    active_hash = manifest.get("active_version_hash")
    if active_hash:
        meta_path = active_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            stored_hash = meta.get("version_hash")
            if stored_hash and stored_hash != active_hash:
                result["status"] = "FAIL"
                result["issues"].append(
                    f"Active version hash mismatch: manifest={active_hash}, meta={stored_hash}"
                )

    return result


def main():
    parser = argparse.ArgumentParser(description="Registry integrity audit")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir)

    print("Registry Audit Report")
    print("=" * 60)

    # Find all versions
    if not registry_dir.exists():
        print("No versions directory found.")
        sys.exit(1)

    version_dirs = sorted(
        d for d in registry_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    print(f"Versions found: {len(version_dirs)}")

    # Audit manifest
    manifest_result = audit_manifest(registry_dir)
    print(f"Active champion: {manifest_result['active_version'] or '(none)'}")
    print()

    # Audit each version
    all_pass = True
    for vdir in version_dirs:
        result = audit_version(vdir)
        files_str = f"{result['files_present']}/{result['files_required']} files"
        status = result["status"]

        if status == "PASS":
            detail = f"{files_str}, all checks passed"
        elif status == "WARN":
            detail = f"{files_str}, {'; '.join(result['issues'])}"
        else:
            detail = f"{files_str}, {'; '.join(result['issues'])}"
            all_pass = False

        print(f"  {result['version']:40s} ... {status:4s} ({detail})")

    # Manifest result
    print()
    if manifest_result["issues"]:
        print(f"Manifest integrity: {manifest_result['status']} ({'; '.join(manifest_result['issues'])})")
    else:
        print(f"Manifest integrity: {manifest_result['status']}")

    print()
    if all_pass and manifest_result["status"] != "FAIL":
        print("Overall: PASS")
    else:
        print("Overall: ISSUES FOUND — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
