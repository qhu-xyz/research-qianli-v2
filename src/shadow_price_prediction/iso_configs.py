from dataclasses import dataclass


@dataclass
class HorizonGroupConfig:
    """Configuration for a specific forecast horizon group."""

    name: str
    min_horizon: int
    max_horizon: int
    description: str = ""
    weight_config_name: str = "short_term"  # 'short_term' or 'long_term'


@dataclass
class DataPathConfig:
    """Data path templates."""

    density_path_template: str
    constraint_path_template: str


@dataclass
class IsoConfig:
    """ISO-specific configuration."""

    name: str
    ap_tools_class: str  # Import path or class name
    auction_schedule: dict[int, list[str]]
    data_paths: DataPathConfig
    outage_freq: str = "3D"
    run_at_day: int = 10


# MISO Configuration
MISO_ISO_CONFIG = IsoConfig(
    name="MISO",
    ap_tools_class="pbase.analysis.tools.all_positions.MisoApTools",
    auction_schedule={
        6: ["f0"],
        7: ["f0", "f1", "q2", "q3", "q4"],
        8: ["f0", "f1", "f2", "f3"],
        9: ["f0", "f1", "f2"],
        10: ["f0", "f1", "q3", "q4"],
        11: ["f0", "f1", "f2", "f3"],
        12: ["f0", "f1", "f2"],
        1: ["f0", "f1", "q4"],
        2: ["f0", "f1", "f2", "f3"],
        3: ["f0", "f1", "f2"],
        4: ["f0", "f1"],
        5: ["f0"],
    },
    data_paths=DataPathConfig(
        density_path_template=(
            "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/"
            "auction_month={auction_month}/market_month={market_month}/"
            "market_round={market_round}/outage_date={outage_date}"
        ),
        constraint_path_template=(
            "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/"
            "auction_month={auction_month}/market_round={market_round}/"
            "period_type={period_type}/class_type={class_type}"
        ),
    ),
    run_at_day=10,
)

MISO_HORIZON_GROUPS = [
    HorizonGroupConfig("f0", 0, 0, "Current Month", "short_term"),
    HorizonGroupConfig("f1", 1, 1, "Next Month", "short_term"),
    HorizonGroupConfig("long", 2, 999, "Long Term (3+ months)", "long_term"),
]

# PJM Configuration
PJM_ISO_CONFIG = IsoConfig(
    name="PJM",
    ap_tools_class="pbase.analysis.tools.all_positions.PjmApTools",
    auction_schedule={
        6: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
        7: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
        8: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
        9: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
        10: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
        11: ["f0", "f1", "f2", "f3", "f4", "f5", "f6"],
        12: ["f0", "f1", "f2", "f3", "f4", "f5"],
        1: ["f0", "f1", "f2", "f3", "f4"],
        2: ["f0", "f1", "f2", "f3"],
        3: ["f0", "f1", "f2"],
        4: ["f0", "f1"],
        5: ["f0"],
    },
    data_paths=DataPathConfig(
        density_path_template=(
            "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density/"
            "auction_month={auction_month}/market_month={market_month}/"
            "market_round={market_round}/outage_date={outage_date}"
        ),
        constraint_path_template=(
            "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info/"
            "auction_month={auction_month}/market_round={market_round}/"
            "period_type={period_type}/class_type={class_type}"
        ),
    ),
    run_at_day=13,
)

PJM_HORIZON_GROUPS = [
    HorizonGroupConfig("f0", 0, 0, "Current Month", "short_term"),
    HorizonGroupConfig("f1", 1, 1, "Next Month", "short_term"),
    HorizonGroupConfig("f2", 2, 2, "Month 2", "long_term"),
    HorizonGroupConfig("long", 3, 999, "Long Term (3+ months)", "long_term"),
]
