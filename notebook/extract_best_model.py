import pandas as pd
from typing import Union, Dict, Any
from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.tuning_utils import update_config_with_params
from shadow_price_prediction.pipeline import ShadowPricePipeline

def create_pipeline_from_row(
    params_row: Union[pd.Series, Dict[str, Any]]
) -> ShadowPricePipeline:
    """
    Creates a ShadowPricePipeline using parameters from a specific result row.
    
    Args:
        params_row: A pandas Series (row from results DataFrame) or a dictionary 
                    containing the parameters.
        
    Returns:
        pipeline: Initialized ShadowPricePipeline with the specified parameters.
    """
    # Convert Series to dict if necessary
    if isinstance(params_row, pd.Series):
        params_dict = params_row.to_dict()
    else:
        params_dict = params_row
        
    print("Initializing pipeline with selected parameters:")
    for k, v in params_dict.items():
        if k.startswith(('clf_', 'reg_', 'threshold_')):
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
