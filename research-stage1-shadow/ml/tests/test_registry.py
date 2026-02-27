import json

import pytest

from ml.registry import allocate_version_id, get_champion, promote_version, register_version


def test_allocate_increments(tmp_path):
    """allocate_version_id increments atomically."""
    counter = tmp_path / "version_counter.json"
    counter.write_text('{"next_id": 1}')
    v1 = allocate_version_id(counter)
    v2 = allocate_version_id(counter)
    assert v1 == "v0001"
    assert v2 == "v0002"


def test_allocate_format(tmp_path):
    """Version IDs are zero-padded to 4 digits."""
    counter = tmp_path / "version_counter.json"
    counter.write_text('{"next_id": 42}')
    vid = allocate_version_id(counter)
    assert vid == "v0042"


def test_register_creates_directory(tmp_path):
    """register_version creates version directory with all JSON files."""
    reg = tmp_path / "registry"
    reg.mkdir()
    register_version(reg, "v0001", {"param": 1}, {"S1-AUC": 0.7}, {"created": "now"})
    assert (reg / "v0001" / "metrics.json").exists()
    assert (reg / "v0001" / "config.json").exists()
    assert (reg / "v0001" / "meta.json").exists()


def test_register_metrics_content(tmp_path):
    """Registered metrics should be readable and correct."""
    reg = tmp_path / "registry"
    reg.mkdir()
    register_version(reg, "v0001", {}, {"S1-AUC": 0.7, "S1-AP": 0.15}, {})
    with open(reg / "v0001" / "metrics.json") as f:
        data = json.load(f)
    assert data["S1-AUC"] == 0.7
    assert data["S1-AP"] == 0.15


def test_register_fails_on_duplicate(tmp_path):
    """register_version should fail if directory already exists."""
    reg = tmp_path / "registry"
    reg.mkdir()
    register_version(reg, "v0001", {}, {}, {})
    with pytest.raises(FileExistsError):
        register_version(reg, "v0001", {}, {}, {})


def test_promote_and_get_champion(tmp_path):
    """promote_version updates champion.json."""
    reg = tmp_path / "registry"
    reg.mkdir()
    champion_path = reg / "champion.json"
    champion_path.write_text('{"version": null, "promoted_at": null}')

    assert get_champion(champion_path) is None

    promote_version(reg, "v0001", champion_path)
    assert get_champion(champion_path) == "v0001"


def test_get_champion_missing_file(tmp_path):
    """get_champion returns None if file doesn't exist."""
    assert get_champion(tmp_path / "nonexistent.json") is None


def test_register_with_model_file(tmp_path):
    """register_version copies model file when provided."""
    reg = tmp_path / "registry"
    reg.mkdir()
    model_file = tmp_path / "model.json"
    model_file.write_text('{"tree": "data"}')

    register_version(reg, "v0001", {}, {}, {}, model_path=model_file)
    assert (reg / "v0001" / "model" / "model.json").exists()


# --- populate_v0_gates tests ---


def test_populate_v0_gates_brier_direction(tmp_path):
    """BRIER is lower-direction: floor = v0 + offset, NOT v0 - offset."""
    from ml.populate_v0_gates import populate_v0_gates

    reg = tmp_path / "registry"
    reg.mkdir()
    v0 = reg / "v0"
    v0.mkdir()
    (v0 / "metrics.json").write_text(json.dumps({
        "S1-AUC": 0.75,
        "S1-AP": 0.15,
        "S1-VCAP@100": 0.50,
        "S1-VCAP@500": 0.60,
        "S1-VCAP@1000": 0.70,
        "S1-NDCG": 0.80,
        "S1-BRIER": 0.089,
        "S1-REC": 0.55,
        "S1-CAP@100": 0.45,
        "S1-CAP@500": 0.55,
    }))

    gates_path = reg / "gates.json"
    gates_data = {
        "version": 1,
        "noise_tolerance": 0.02,
        "gates": {
            "S1-AUC": {"floor": 0.65, "direction": "higher", "pending_v0": False},
            "S1-AP": {"floor": 0.12, "direction": "higher", "pending_v0": False},
            "S1-VCAP@100": {"floor": None, "direction": "higher", "pending_v0": True, "v0_offset": 0.05},
            "S1-BRIER": {"floor": None, "direction": "lower", "pending_v0": True, "v0_offset": 0.02},
            "S1-REC": {"floor": 0.40, "direction": "higher", "pending_v0": False, "group": "B"},
        },
    }
    gates_path.write_text(json.dumps(gates_data))

    populate_v0_gates(str(reg), str(gates_path))

    with open(gates_path) as f:
        updated = json.load(f)

    # BRIER: lower direction => floor = v0 + offset = 0.089 + 0.02 = 0.109
    assert abs(updated["gates"]["S1-BRIER"]["floor"] - 0.109) < 1e-6
    assert updated["gates"]["S1-BRIER"]["pending_v0"] is False

    # VCAP@100: higher direction => floor = v0 - offset = 0.50 - 0.05 = 0.45
    assert abs(updated["gates"]["S1-VCAP@100"]["floor"] - 0.45) < 1e-6
    assert updated["gates"]["S1-VCAP@100"]["pending_v0"] is False

    # Non-pending gates should not change
    assert updated["gates"]["S1-AUC"]["floor"] == 0.65
    assert updated["gates"]["S1-AUC"].get("pending_v0") is False


def test_populate_v0_gates_idempotent(tmp_path):
    """Running populate twice is a no-op the second time."""
    from ml.populate_v0_gates import populate_v0_gates

    reg = tmp_path / "registry"
    reg.mkdir()
    v0 = reg / "v0"
    v0.mkdir()
    (v0 / "metrics.json").write_text(json.dumps({
        "S1-BRIER": 0.089,
        "S1-VCAP@100": 0.50,
    }))

    gates_data = {
        "version": 1,
        "noise_tolerance": 0.02,
        "gates": {
            "S1-BRIER": {"floor": None, "direction": "lower", "pending_v0": True, "v0_offset": 0.02},
            "S1-VCAP@100": {"floor": None, "direction": "higher", "pending_v0": True, "v0_offset": 0.05},
        },
    }
    gates_path = reg / "gates.json"
    gates_path.write_text(json.dumps(gates_data))

    populate_v0_gates(str(reg), str(gates_path))
    # Read the result after first run
    with open(gates_path) as f:
        first_run = json.load(f)

    # Run again -- should be no-op
    populate_v0_gates(str(reg), str(gates_path))
    with open(gates_path) as f:
        second_run = json.load(f)

    assert first_run == second_run
