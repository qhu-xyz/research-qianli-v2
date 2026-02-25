"""
Model version registry for shadow price prediction pipeline.

Manages versioning, gating, promotion, and comparison of model configurations
and their benchmark results. Each version is stored as a directory of JSON files
with a config checksum for integrity verification.

Usage
-----
>>> from shadow_price_prediction.registry import ModelRegistry
>>> reg = ModelRegistry("versions/")
>>> ver = reg.create_version(algo="xgb", config=my_config, description="Added hist_da")
>>> ver.record_metrics(scores, gate_values)
>>> result = reg.check_gates(ver.model_id)
>>> if result.all_passed:
...     reg.promote(ver.model_id)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

# Each gate specifies:
#   metric_path : dot-separated path into gate_values dict
#   direction   : "higher" (floor) or "lower" (ceiling)
#   floor/ceil  : absolute threshold (None = tracked but not enforced)

HARD_GATES: dict[str, dict[str, Any]] = {
    "S1-AUC": {
        "description": "Stage 1 AUC-ROC",
        "direction": "higher",
        "floor": 0.65,
    },
    "S1-REC": {
        "description": "Stage 1 Recall",
        "direction": "higher",
        "floor": 0.25,
    },
    "S1-PREC": {
        "description": "Stage 1 Precision",
        "direction": "higher",
        "floor": 0.25,
    },
    "S2-SPR": {
        "description": "Stage 2 Spearman (TP)",
        "direction": "higher",
        "floor": 0.30,
    },
    "C-VC@100": {
        "description": "Value Capture top-100 constraints",
        "direction": "higher",
        "floor": 0.20,
    },
    "C-VC@500": {
        "description": "Value Capture top-500 constraints",
        "direction": "higher",
        "floor": 0.45,
    },
    "C-VC@1000": {
        "description": "Value Capture top-1000 constraints",
        "direction": "higher",
        "floor": 0.50,
    },
    "C-CAP@20": {
        "description": "Capture rate top-20 constraints",
        "direction": "higher",
        "floor": 0.50,
    },
    "C-CAP@200": {
        "description": "Capture rate top-200 constraints",
        "direction": "higher",
        "floor": 0.30,
    },
    "C-CAP@1000": {
        "description": "Capture rate top-1000 constraints",
        "direction": "higher",
        "floor": 0.10,
    },
    "C-RMSE": {
        "description": "Combined RMSE (all)",
        "direction": "lower",
        "floor": 2000.0,
    },
}

# Noise tolerance for promotion: candidate must be >= champion * (1 - TOLERANCE)
# for "higher is better" gates, and <= champion * (1 + TOLERANCE) for "lower is better".
# Additionally, at least one gate must show > TOLERANCE improvement.
NOISE_TOLERANCE = 0.02


def _compute_checksum(data: dict) -> str:
    """Deterministic SHA-256 checksum of a JSON-serializable dict."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GateCheck:
    """Result of checking one gate."""

    gate: str
    description: str
    direction: str
    threshold: float | None
    value: float
    passed: bool
    champion_value: float | None = None
    beats_champion: bool | None = None
    failed_segments: list[str] = field(default_factory=list)  # e.g. ["onpeak/f1"]


@dataclass
class GateResult:
    """Aggregate result of all gate checks."""

    model_id: str
    champion_id: str | None
    checks: list[GateCheck]
    timestamp: str = field(default_factory=_now_iso)

    @property
    def all_passed(self) -> bool:
        """True if every gate with a threshold passes."""
        return all(c.passed for c in self.checks if c.threshold is not None)

    @property
    def beats_champion(self) -> bool:
        """True if candidate beats champion within noise tolerance on all enforced gates,
        AND at least one gate shows meaningful improvement (> NOISE_TOLERANCE).

        This prevents promoting a model that is merely "not worse" — it must
        actually improve something.
        """
        if self.champion_id is None:
            return True
        enforced = [c for c in self.checks if c.threshold is not None]
        # All enforced gates must be within tolerance
        all_within_tolerance = all(
            c.beats_champion is True or c.beats_champion is None
            for c in enforced
        )
        if not all_within_tolerance:
            return False
        # At least one gate must show genuine improvement beyond tolerance
        has_improvement = False
        for c in enforced:
            if c.champion_value is not None and not np.isnan(c.value):
                if c.direction == "higher":
                    if c.value > c.champion_value * (1 + NOISE_TOLERANCE):
                        has_improvement = True
                        break
                else:
                    if c.value < c.champion_value * (1 - NOISE_TOLERANCE):
                        has_improvement = True
                        break
        return has_improvement

    @property
    def promotable(self) -> bool:
        """True if all gates pass AND beats champion."""
        return self.all_passed and self.beats_champion

    def summary_table(self) -> str:
        """Return a formatted summary table."""
        lines = [
            f"Gate check for {self.model_id} vs champion {self.champion_id or '(none)'}",
            f"{'Gate':<12} {'Value':>8} {'Floor':>8} {'Pass':>6} {'Champ':>8} {'Beats':>6}  {'Failed Segments'}",
            "-" * 72,
        ]
        for c in self.checks:
            thr_str = f"{c.threshold:.4f}" if c.threshold is not None else "   n/a"
            champ_str = f"{c.champion_value:.4f}" if c.champion_value is not None else "   n/a"
            beats_str = "Y" if c.beats_champion else ("N" if c.beats_champion is False else "-")
            pass_str = "PASS" if c.passed else "FAIL"
            fail_str = ", ".join(c.failed_segments) if c.failed_segments else ""
            lines.append(
                f"{c.gate:<12} {c.value:>8.4f} {thr_str:>8} {pass_str:>6} {champ_str:>8} {beats_str:>6}  {fail_str}"
            )
        lines.append("-" * 72)
        lines.append(f"All passed: {self.all_passed}  |  Beats champion: {self.beats_champion}  |  Promotable: {self.promotable}")
        return "\n".join(lines)


@dataclass
class PromotionResult:
    """Result of a promotion attempt."""

    success: bool
    model_id: str
    reason: str
    previous_champion: str | None = None


# ---------------------------------------------------------------------------
# ModelVersion — one versioned experiment
# ---------------------------------------------------------------------------


class ModelVersion:
    """A single model version stored on disk.

    Directory layout::

        versions/{model_id}/
            meta.json                — id, date, commit, checksum, version_hash, status
            config.json              — full PredictionConfig
            features.json            — feature lists + monotonicity constraints
            metrics.json             — gate_values + per_run_scores + benchmark_scope
            threshold_manifest.json  — per-run threshold decisions
            feature_importance.json  — per-model feature importances (diagnostic only)
            train_manifest.json      — training data provenance
    """

    def __init__(self, version_dir: Path):
        self._dir = version_dir
        self._meta: dict | None = None

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def model_id(self) -> str:
        return self._dir.name

    # ---- Lazy loaders ----

    def _load_json(self, name: str) -> dict:
        path = self._dir / name
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_json(self, name: str, data: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._dir / name, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = self._load_json("meta.json")
        return self._meta

    @property
    def config(self) -> dict:
        return self._load_json("config.json")

    @property
    def features(self) -> dict:
        return self._load_json("features.json")

    @property
    def metrics(self) -> dict:
        return self._load_json("metrics.json")

    @property
    def status(self) -> str:
        return self.meta.get("status", "unknown")

    @property
    def config_checksum(self) -> str:
        return self.meta.get("config_checksum", "")

    def verify_checksum(self) -> bool:
        """Verify that config.json matches the checksum in meta.json."""
        expected = self.config_checksum
        if not expected:
            return False
        actual = _compute_checksum(self.config)
        return actual == expected

    # ---- Gate values ----

    def gate_values(self, class_type: str = "onpeak") -> dict[str, float]:
        """Return gate metric values for a class type."""
        m = self.metrics
        gv = m.get("gate_values", {})
        return gv.get(class_type, {})

    def gate_values_by_period(self, class_type: str, period_type: str) -> dict[str, float]:
        """Return gate metric values for a (class_type, period_type) segment."""
        m = self.metrics
        gv = m.get("gate_values_by_period", {})
        key = f"{class_type}/{period_type}"
        return gv.get(key, {})

    # ---- Recording ----

    def record_config(self, config_dict: dict) -> str:
        """Save config.json and return its checksum."""
        self._save_json("config.json", config_dict)
        checksum = _compute_checksum(config_dict)
        # Update meta with checksum
        meta = self.meta
        meta["config_checksum"] = checksum
        self._meta = meta
        self._save_json("meta.json", meta)
        return checksum

    def record_features(self, features_dict: dict) -> None:
        """Save features.json."""
        self._save_json("features.json", features_dict)

    def record_metrics(
        self,
        gate_values: dict[str, dict[str, float]],
        benchmark_scope: dict | None = None,
        per_run_scores: list[dict] | None = None,
        gate_values_by_period: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Save metrics.json with gate values and optional details."""
        data: dict[str, Any] = {"gate_values": gate_values}
        if gate_values_by_period is not None:
            data["gate_values_by_period"] = gate_values_by_period
        if benchmark_scope is not None:
            data["benchmark_scope"] = benchmark_scope
        if per_run_scores is not None:
            data["per_run_scores"] = per_run_scores
        self._save_json("metrics.json", data)

    # ---- New artifact recording (threshold, feature importance, train manifest) ----

    def record_threshold_manifest(self, data: dict) -> None:
        """Save threshold_manifest.json — per-run threshold decisions.

        Expected shape::

            {
              "202007_onpeak": {
                "horizon_group_short": {
                  "default": {"threshold": 0.42, "f_beta_used": 2.0, ...},
                  "branch_X": {...}
                }
              },
              ...
            }
        """
        self._save_json("threshold_manifest.json", data)

    def record_feature_importance(self, data: dict) -> None:
        """Save feature_importance.json — per-model feature importances.

        Expected shape::

            {
              "stage1": {
                "horizon_group": {
                  "default": {"gain": {...}, "cover": {...}, "weight": {...}}
                }
              },
              "stage2": { ... }
            }

        Note: Feature importances are *unstable* across retrains and should
        not be used as model identity verification.  They are stored for
        diagnostic / interpretability purposes only.
        """
        self._save_json("feature_importance.json", data)

    def record_train_manifest(self, data: dict) -> None:
        """Save train_manifest.json — training data provenance.

        Expected shape::

            {
              "202007_onpeak": {
                "auction_month": "2020-07",
                "class_type": "onpeak",
                "training_date_range": ["2019-01", "2020-06"],
                "n_rows_total": 45000,
                "n_binding": 3200,
                "binding_rate": 0.071,
                "feature_stats": { ... }
              },
              ...
            }
        """
        self._save_json("train_manifest.json", data)

    # ---- Version hash (integrity over all artifacts) ----

    def compute_version_hash(self) -> str:
        """Compute a SHA-256 hash over all required artifacts for integrity.

        The hash covers config.json, features.json, metrics.json,
        threshold_manifest.json, feature_importance.json, and train_manifest.json.
        It is stored in meta.json as ``version_hash``.

        Returns
        -------
        str
            The version hash in format ``sha256:<hex>``.
        """
        artifact_names = [
            "config.json",
            "features.json",
            "metrics.json",
            "threshold_manifest.json",
            "feature_importance.json",
            "train_manifest.json",
        ]
        hasher = hashlib.sha256()
        for name in artifact_names:
            data = self._load_json(name)
            if data:
                canonical = json.dumps(data, sort_keys=True, default=str)
                hasher.update(canonical.encode("utf-8"))
            else:
                hasher.update(b"__MISSING__")

        version_hash = "sha256:" + hasher.hexdigest()[:32]

        # Persist in meta.json
        meta = self.meta
        meta["version_hash"] = version_hash
        self._meta = meta
        self._save_json("meta.json", meta)

        return version_hash

    def verify_version_hash(self) -> bool:
        """Recompute version hash and compare to stored value."""
        stored = self.meta.get("version_hash", "")
        if not stored:
            return False
        # Recompute without saving
        artifact_names = [
            "config.json",
            "features.json",
            "metrics.json",
            "threshold_manifest.json",
            "feature_importance.json",
            "train_manifest.json",
        ]
        hasher = hashlib.sha256()
        for name in artifact_names:
            data = self._load_json(name)
            if data:
                canonical = json.dumps(data, sort_keys=True, default=str)
                hasher.update(canonical.encode("utf-8"))
            else:
                hasher.update(b"__MISSING__")
        recomputed = "sha256:" + hasher.hexdigest()[:32]
        return recomputed == stored

    # ---- Status ----

    def set_status(self, status: str) -> None:
        """Update the status field in meta.json."""
        meta = self.meta
        meta["status"] = status
        meta["status_updated"] = _now_iso()
        self._meta = meta
        self._save_json("meta.json", meta)


# ---------------------------------------------------------------------------
# ModelRegistry — manages all versions
# ---------------------------------------------------------------------------


class ModelRegistry:
    """File-based model registry.

    Parameters
    ----------
    registry_dir : str or Path
        Root directory for the registry.  Will be created if it doesn't exist.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(self, registry_dir: str | Path):
        self._root = Path(registry_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._versions_dir = self._root

    # ---- Manifest ----

    def _load_manifest(self) -> dict:
        path = self._root / self.MANIFEST_FILE
        if not path.exists():
            return {"active_version": None, "active_checksum": None, "history": []}
        with open(path) as f:
            return json.load(f)

    def _save_manifest(self, manifest: dict) -> None:
        with open(self._root / self.MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    @property
    def active_version(self) -> str | None:
        """Model ID of the currently active (champion) version."""
        return self._load_manifest().get("active_version")

    def get_champion(self) -> ModelVersion | None:
        """Return the active champion ModelVersion, or None."""
        vid = self.active_version
        if vid is None:
            return None
        return self.get_version(vid)

    # ---- Version management ----

    def list_versions(self) -> list[str]:
        """Return sorted list of all version IDs."""
        if not self._versions_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._versions_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )

    def get_version(self, model_id: str) -> ModelVersion:
        """Get a ModelVersion by ID."""
        d = self._versions_dir / model_id
        if not d.exists():
            raise FileNotFoundError(f"Version {model_id} not found in {self._versions_dir}")
        return ModelVersion(d)

    def _next_seq(self, algo: str, date_str: str) -> int:
        """Determine next sequence number for a given algo+date."""
        pattern = re.compile(rf"v\d{{3}}-{re.escape(algo)}-{date_str}-(\d{{3}})")
        max_seq = 0
        for vid in self.list_versions():
            m = pattern.match(vid)
            if m:
                max_seq = max(max_seq, int(m.group(1)))
        return max_seq + 1

    def _next_version_number(self) -> int:
        """Determine next version number (vNNN)."""
        max_v = -1
        for vid in self.list_versions():
            m = re.match(r"v(\d{3})", vid)
            if m:
                max_v = max(max_v, int(m.group(1)))
        return max_v + 1

    def create_version(
        self,
        algo: str,
        description: str,
        config_dict: dict | None = None,
        features_dict: dict | None = None,
        source_commit: str | None = None,
        model_id_override: str | None = None,
    ) -> ModelVersion:
        """Create a new model version.

        Parameters
        ----------
        algo : str
            Algorithm identifier (e.g. "xgb", "lgbm", "legacy").
        description : str
            Human-readable description of what changed.
        config_dict : dict, optional
            Serialized PredictionConfig.  If provided, saved and checksummed.
        features_dict : dict, optional
            Feature specification.  If provided, saved.
        source_commit : str, optional
            Git commit hash.  Auto-detected if not provided.
        model_id_override : str, optional
            Override the auto-generated model ID (for bootstrapping baseline).

        Returns
        -------
        ModelVersion
        """
        if model_id_override:
            model_id = model_id_override
        else:
            v_num = self._next_version_number()
            date_str = datetime.now().strftime("%Y%m%d")
            seq = self._next_seq(algo, date_str)
            model_id = f"v{v_num:03d}-{algo}-{date_str}-{seq:03d}"

        version_dir = self._versions_dir / model_id
        version_dir.mkdir(parents=True, exist_ok=True)

        commit = source_commit or _get_git_commit()

        meta = {
            "model_id": model_id,
            "created": _now_iso(),
            "description": description,
            "source_commit": commit,
            "algo": algo,
            "status": "candidate",
            "config_checksum": None,
        }

        ver = ModelVersion(version_dir)
        ver._meta = meta
        ver._save_json("meta.json", meta)

        if config_dict is not None:
            ver.record_config(config_dict)

        if features_dict is not None:
            ver.record_features(features_dict)

        return ver

    # ---- Gate checking ----

    def check_gates(
        self,
        model_id: str,
        class_types: tuple[str, ...] = ("onpeak", "offpeak"),
        period_types: tuple[str, ...] = ("f0", "f1"),
        gates: dict[str, dict[str, Any]] | None = None,
    ) -> GateResult:
        """Check a candidate version against hard gates and champion.

        **Per-segment enforcement**: Each (class_type, period_type) segment
        must individually pass every floor gate.  The reported ``value`` is
        the cross-segment average (for display), but ``passed`` is False if
        *any* individual segment fails.  ``failed_segments`` lists which
        segments failed (e.g. ``["onpeak/f1"]``).

        **Noise-tolerant champion comparison**: A candidate "beats" the champion
        if ``cand >= champ * (1 - NOISE_TOLERANCE)`` (higher-is-better) or
        ``cand <= champ * (1 + NOISE_TOLERANCE)`` (lower-is-better).  This
        prevents random variance from blocking legitimate improvements.

        Parameters
        ----------
        model_id : str
            Candidate version to check.
        class_types : tuple of str
            Class types to enforce.
        period_types : tuple of str
            Period types to enforce.  Each (ct, pt) combo is checked.
        gates : dict, optional
            Gate definitions.  Defaults to ``HARD_GATES``.

        Returns
        -------
        GateResult
        """
        if gates is None:
            gates = HARD_GATES

        candidate = self.get_version(model_id)
        champion = self.get_champion()

        # Build list of segments to check
        segments = [(ct, pt) for ct in class_types for pt in period_types]

        checks: list[GateCheck] = []

        for gate_name, gate_def in gates.items():
            direction = gate_def["direction"]
            threshold = gate_def.get("floor")

            # Collect per-segment candidate values and check floors
            cand_vals = []
            failed_segs: list[str] = []

            for ct, pt in segments:
                # Try per-period values first, fall back to class-level
                gv = candidate.gate_values_by_period(ct, pt)
                if gate_name not in gv:
                    # Fallback: use class-level values (old versions without per-period data)
                    gv = candidate.gate_values(ct)

                if gate_name in gv:
                    v = gv[gate_name]
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        cand_vals.append(v)
                        # Per-segment floor enforcement
                        if threshold is not None:
                            if direction == "higher" and v < threshold:
                                failed_segs.append(f"{ct}/{pt}")
                            elif direction == "lower" and v > threshold:
                                failed_segs.append(f"{ct}/{pt}")

            cand_value = float(np.mean(cand_vals)) if cand_vals else float("nan")

            # Gate passes only if ALL segments pass
            if threshold is not None and not np.isnan(cand_value):
                passed = len(failed_segs) == 0
            elif threshold is None:
                passed = True
            else:
                passed = False  # NaN value fails

            # Compare to champion (with noise tolerance)
            champ_value: float | None = None
            beats: bool | None = None
            if champion is not None:
                champ_vals = []
                for ct, pt in segments:
                    gv = champion.gate_values_by_period(ct, pt)
                    if gate_name not in gv:
                        gv = champion.gate_values(ct)
                    if gate_name in gv:
                        v = gv[gate_name]
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            champ_vals.append(v)
                if champ_vals:
                    champ_value = float(np.mean(champ_vals))
                    if not np.isnan(cand_value):
                        if direction == "higher":
                            beats = cand_value >= champ_value * (1 - NOISE_TOLERANCE)
                        else:
                            beats = cand_value <= champ_value * (1 + NOISE_TOLERANCE)

            checks.append(
                GateCheck(
                    gate=gate_name,
                    description=gate_def["description"],
                    direction=direction,
                    threshold=threshold,
                    value=cand_value,
                    passed=passed,
                    champion_value=champ_value,
                    beats_champion=beats,
                    failed_segments=failed_segs,
                )
            )

        return GateResult(
            model_id=model_id,
            champion_id=champion.model_id if champion else None,
            checks=checks,
        )

    # ---- Promotion ----

    def promote(
        self,
        model_id: str,
        force: bool = False,
        reason: str | None = None,
    ) -> PromotionResult:
        """Promote a version to active champion.

        Parameters
        ----------
        model_id : str
            Version to promote.
        force : bool
            If True, skip gate checks (for bootstrapping baseline).
        reason : str, optional
            Override reason string.

        Returns
        -------
        PromotionResult
        """
        candidate = self.get_version(model_id)

        if not force:
            gate_result = self.check_gates(model_id)
            if not gate_result.promotable:
                return PromotionResult(
                    success=False,
                    model_id=model_id,
                    reason=f"Gate check failed:\n{gate_result.summary_table()}",
                )

        # Verify config checksum
        if candidate.config and candidate.config_checksum:
            if not candidate.verify_checksum():
                return PromotionResult(
                    success=False,
                    model_id=model_id,
                    reason="Config checksum verification failed. Config may have been tampered with.",
                )

        manifest = self._load_manifest()
        previous = manifest.get("active_version")

        # Demote previous champion
        if previous:
            try:
                prev_ver = self.get_version(previous)
                prev_ver.set_status("retired")
                # Record demotion in history
                for entry in manifest.get("history", []):
                    if entry.get("version") == previous and entry.get("demoted_at") is None:
                        entry["demoted_at"] = _now_iso()
            except FileNotFoundError:
                pass

        # Promote candidate
        candidate.set_status("active")
        manifest["active_version"] = model_id
        manifest["active_checksum"] = candidate.config_checksum
        manifest["active_version_hash"] = candidate.meta.get("version_hash")
        manifest["promoted_at"] = _now_iso()
        manifest.setdefault("history", []).append(
            {
                "version": model_id,
                "promoted_at": _now_iso(),
                "demoted_at": None,
            }
        )
        self._save_manifest(manifest)

        return PromotionResult(
            success=True,
            model_id=model_id,
            reason=reason or "Passed all gates and beats champion",
            previous_champion=previous,
        )

    # ---- Comparison ----

    def compare(
        self,
        version_a: str,
        version_b: str,
        class_types: tuple[str, ...] = ("onpeak", "offpeak"),
    ) -> str:
        """Compare two versions side by side. Returns formatted table."""
        va = self.get_version(version_a)
        vb = self.get_version(version_b)

        lines = [
            f"Comparison: {version_a} vs {version_b}",
            f"{'':15} {'':>5} {version_a:>20} {version_b:>20} {'Delta':>10}",
            "-" * 75,
        ]

        for ct in class_types:
            gv_a = va.gate_values(ct)
            gv_b = vb.gate_values(ct)
            lines.append(f"\n  {ct}:")
            all_keys = sorted(set(list(gv_a.keys()) + list(gv_b.keys())))
            for key in all_keys:
                val_a = gv_a.get(key, float("nan"))
                val_b = gv_b.get(key, float("nan"))
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    delta = val_b - val_a
                    delta_str = f"{delta:+.4f}"
                else:
                    delta_str = "n/a"
                lines.append(
                    f"  {key:<15} {val_a:>20} {val_b:>20} {delta_str:>10}"
                )

        return "\n".join(lines)

    # ---- Integrity ----

    def verify_active(self) -> bool:
        """Verify that the active version's config matches the manifest checksum."""
        manifest = self._load_manifest()
        vid = manifest.get("active_version")
        expected_checksum = manifest.get("active_checksum")
        if vid is None or expected_checksum is None:
            return False
        try:
            ver = self.get_version(vid)
            actual_checksum = _compute_checksum(ver.config)
            return actual_checksum == expected_checksum
        except FileNotFoundError:
            return False

    # ---- Serialization helpers ----

    @staticmethod
    def config_to_dict(config) -> dict:
        """Serialize a PredictionConfig dataclass to a JSON-safe dict.

        Handles model_class references by storing their qualified name.
        """
        from dataclasses import asdict, fields, is_dataclass

        def _convert(obj: Any) -> Any:
            if is_dataclass(obj):
                result = {}
                for f in fields(obj):
                    val = getattr(obj, f.name)
                    result[f.name] = _convert(val)
                return result
            elif isinstance(obj, type):
                # Model class → qualified name string
                return f"{obj.__module__}.{obj.__qualname__}"
            elif isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return _convert(config)

    @staticmethod
    def features_to_dict(config) -> dict:
        """Extract feature specification from a PredictionConfig."""
        fc = config.features
        return {
            "step1_features": [
                {"name": name, "monotonicity": mono}
                for name, mono in fc.step1_features
            ],
            "step2_features": [
                {"name": name, "monotonicity": mono}
                for name, mono in fc.step2_features
            ],
            "all_features": fc.all_features,
        }
