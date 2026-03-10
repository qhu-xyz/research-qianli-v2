import pandas as pd
from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.tuning_utils import update_config_with_params
from shadow_price_prediction.pipeline import ShadowPricePipeline


def create_pipeline_from_row(row):
    """
    Create a ShadowPricePipeline using the parameters from a single row of results.

    Args:
        row: DataFrame row with parameters

    Returns:
        ShadowPricePipeline configured with the parameters
    """
    params_dict = row.to_dict()

    print("Initializing pipeline with selected parameters:")
    for k, v in params_dict.items():
        if k.startswith(("clf_", "reg_", "threshold_", "label_")):
            print(f"  {k}: {v}")

    # Create Config
    config = PredictionConfig()

    # Update Config with Params
    config = update_config_with_params(config, params_dict)

    # Initialize Pipeline
    pipeline = ShadowPricePipeline(config)

    return pipeline


# Example Usage:
# # Assuming 'results_df' is your dataframe of search results
# selected_row = results_df.iloc[0]
# pipeline = create_pipeline_from_row(selected_row)
#
# # Then run with test_periods:
# test_periods = [(pd.Timestamp('2025-07-01'), pd.Timestamp('2025-08-01'))]
# results_per_outage, final_results, metrics = pipeline.run(test_periods=test_periods)
