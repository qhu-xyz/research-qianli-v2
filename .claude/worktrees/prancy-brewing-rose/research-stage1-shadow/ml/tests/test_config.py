from ml.config import FeatureConfig, HyperparamConfig, PipelineConfig, GateConfig


def test_feature_config_has_14_features():
    fc = FeatureConfig()
    assert len(fc.features) == 14


def test_monotone_constraints_length():
    fc = FeatureConfig()
    mc = fc.get_monotone_constraints_str()
    # String format: "(1,1,1,1,1,...)" -- count commas + 1 == 14
    values = mc.strip("()").split(",")
    assert len(values) == 14


def test_monotone_constraints_values():
    fc = FeatureConfig()
    mc = fc.get_monotone_constraints_str()
    assert mc == "(1,1,1,1,1,-1,-1,-1,1,0,0,0,1,1)"


def test_feature_names_match_expected():
    fc = FeatureConfig()
    expected = [
        "prob_exceed_110",
        "prob_exceed_105",
        "prob_exceed_100",
        "prob_exceed_95",
        "prob_exceed_90",
        "prob_below_100",
        "prob_below_95",
        "prob_below_90",
        "expected_overload",
        "density_skewness",
        "density_kurtosis",
        "density_cv",
        "hist_da",
        "hist_da_trend",
    ]
    assert fc.features == expected


def test_hyperparam_defaults():
    hc = HyperparamConfig()
    assert hc.n_estimators == 200
    assert hc.max_depth == 4
    assert hc.learning_rate == 0.1
    assert hc.subsample == 0.8
    assert hc.colsample_bytree == 0.8
    assert hc.reg_alpha == 0.1
    assert hc.reg_lambda == 1.0
    assert hc.min_child_weight == 10
    assert hc.random_state == 42


def test_hyperparam_to_dict():
    hc = HyperparamConfig()
    d = hc.to_dict()
    assert d["n_estimators"] == 200
    assert d["max_depth"] == 4
    assert len(d) == 9


def test_pipeline_config_defaults():
    pc = PipelineConfig()
    assert pc.threshold_beta == 0.7
    assert pc.train_months == 10
    assert pc.val_months == 2
    assert pc.class_type == "onpeak"
    assert pc.period_type == "f0"
    assert pc.scale_pos_weight_auto is True
    assert pc.registry_dir == "registry"


def test_gate_config_loads(tmp_path):
    gates_file = tmp_path / "gates.json"
    gates_file.write_text(
        '{"version": 1, "noise_tolerance": 0.02, "gates": {"S1-AUC": {"floor": 0.65, "direction": "higher", "pending_v0": false}}}'
    )
    gc = GateConfig(gates_path=str(gates_file))
    assert gc.all_floors_populated() is True
    assert gc.noise_tolerance == 0.02


def test_gate_config_v2_fields(tmp_path):
    """GateConfig v2 must expose cascade_stages, tail fields, and eval_months."""
    import json
    gates_data = {
        "version": 2,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": ["2020-09"], "stress": ["2021-02"]},
        "cascade_stages": [
            {"stage": 1, "ptype": "f0", "blocking": True}
        ],
        "gates": {
            "S1-AUC": {
                "floor": 0.65, "tail_floor": 0.55,
                "direction": "higher", "group": "A",
                "pending_v0": False
            }
        }
    }
    p = tmp_path / "gates.json"
    p.write_text(json.dumps(gates_data))
    gc = GateConfig(str(p))
    assert gc.data["version"] == 2
    assert gc.tail_max_failures == 1
    assert gc.eval_months["primary"] == ["2020-09"]
    assert gc.cascade_stages[0]["ptype"] == "f0"


def test_gate_config_pending_v0(tmp_path):
    gates_file = tmp_path / "gates.json"
    gates_file.write_text(
        '{"version": 1, "noise_tolerance": 0.02, "gates": {"S1-AUC": {"floor": 0.65, "direction": "higher", "pending_v0": false}, "S1-BRIER": {"floor": null, "direction": "lower", "pending_v0": true, "v0_offset": 0.02}}}'
    )
    gc = GateConfig(gates_path=str(gates_file))
    assert gc.all_floors_populated() is False
    assert gc.pending_v0_gates() == ["S1-BRIER"]
