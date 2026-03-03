"""
Canonical path and file-name generators for the shadow price prediction pipeline.

All functions are pure — they take directory roots as arguments and return Path objects.
No hardcoded environment-specific paths exist in this module.
"""

from pathlib import Path


# ── Result parquets ──────────────────────────────────────────────────────

def result_parquet_name(auction_month: str, class_type: str, period_type: str) -> str:
    """e.g. 'results_202007_onpeak_f0.parquet'"""
    am_compact = auction_month.replace("-", "")
    return f"results_{am_compact}_{class_type}_{period_type}.parquet"


def result_parquet_path(
    output_dir: str, auction_month: str, class_type: str, period_type: str
) -> Path:
    return Path(output_dir) / result_parquet_name(auction_month, class_type, period_type)


# ── Registry paths ───────────────────────────────────────────────────────

def version_dir(registry_root: str, model_id: str) -> Path:
    return Path(registry_root) / "versions" / model_id


def config_path(ver_dir: Path) -> Path:
    return ver_dir / "config.json"


def features_path(ver_dir: Path) -> Path:
    return ver_dir / "features.json"


def metrics_path(ver_dir: Path) -> Path:
    return ver_dir / "metrics.json"


def meta_path(ver_dir: Path) -> Path:
    return ver_dir / "meta.json"


def threshold_manifest_path(ver_dir: Path) -> Path:
    return ver_dir / "threshold_manifest.json"


def feature_importance_path(ver_dir: Path) -> Path:
    return ver_dir / "feature_importance.json"


def train_manifest_path(ver_dir: Path) -> Path:
    return ver_dir / "train_manifest.json"


# ── Experiment output ────────────────────────────────────────────────────

def experiment_output_dir(base_dir: str, model_id: str) -> Path:
    return Path(base_dir) / model_id


def agg_csv_path(output_dir: str, model_id: str) -> Path:
    return Path(output_dir) / f"{model_id}_agg.csv"


# ── Artifact metadata paths inside experiment output ─────────────────────

def worker_threshold_path(output_dir: str, auction_month: str, class_type: str) -> Path:
    """Per-run threshold decisions saved by worker, later aggregated."""
    am_compact = auction_month.replace("-", "")
    return Path(output_dir) / f"thresholds_{am_compact}_{class_type}.json"


def worker_feature_importance_path(
    output_dir: str, auction_month: str, class_type: str
) -> Path:
    """Per-run feature importances saved by worker, later aggregated."""
    am_compact = auction_month.replace("-", "")
    return Path(output_dir) / f"feature_importance_{am_compact}_{class_type}.json"


def worker_train_manifest_path(
    output_dir: str, auction_month: str, class_type: str
) -> Path:
    """Per-run training data provenance saved by worker, later aggregated."""
    am_compact = auction_month.replace("-", "")
    return Path(output_dir) / f"train_manifest_{am_compact}_{class_type}.json"


# ── Required registry artifacts ──────────────────────────────────────────

REQUIRED_VERSION_FILES = [
    "meta.json",
    "config.json",
    "features.json",
    "metrics.json",
    "threshold_manifest.json",
    "feature_importance.json",
    "train_manifest.json",
]
