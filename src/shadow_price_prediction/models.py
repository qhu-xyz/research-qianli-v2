"""
Model training for classification and regression with ensemble support.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Set, List, Any
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import fbeta_score

from sklearn.model_selection import cross_val_predict, GroupKFold, TimeSeriesSplit

from .config import PredictionConfig, ModelSpec

class StackingModel:
    """
    Implements Stacking Ensemble (3.B).
    Trains base models and a meta-model on their predictions.
    """
    def __init__(self, base_models: List[Any], meta_model: Any = None, use_proba: bool = True, cv: int = 3, split_by_groups: bool = False):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.use_proba = use_proba
        self.cv = cv
        self.split_by_groups = split_by_groups
        self.is_fitted = False

    def fit(self, X, y, groups=None):
        # 1. Train base models
        # To avoid overfitting, we should ideally use cross-val predictions for the meta-model training
        # But for simplicity in this custom implementation, we'll use the standard approach:
        # Fit base models on full data, but for meta-model training, we need out-of-fold predictions.
        
        meta_features = []
        
        # Determine CV strategy
        cv = self.cv
        if self.split_by_groups and groups is not None:
            # Use GroupKFold if groups are provided and requested
            cv = GroupKFold(n_splits=self.cv)
        
        for model in self.base_models:
            # Get out-of-fold predictions
            if self.use_proba and hasattr(model, "predict_proba"):
                preds = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
            else:
                preds = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict")
            
            meta_features.append(preds.reshape(-1, 1))
            
            # Fit the model on full data for final inference
            model.fit(X, y)
            
        meta_X = np.hstack(meta_features)
        
        # 2. Train meta-model
        self.meta_model.fit(meta_X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        meta_features = []
        for model in self.base_models:
            if self.use_proba and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds.reshape(-1, 1))
            
        meta_X = np.hstack(meta_features)
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        meta_features = []
        for model in self.base_models:
            if self.use_proba and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds.reshape(-1, 1))
            
        meta_X = np.hstack(meta_features)
        return self.meta_model.predict_proba(meta_X)



def create_model(model_spec: ModelSpec, is_classifier: bool = True, scale_pos_weight: float = None) -> Any:
    """
    Factory function to create a model instance based on ModelSpec.

    Parameters:
    -----------
    model_spec : ModelSpec
        Model specification containing model class and configuration
    is_classifier : bool
        Whether creating a classifier (True) or regressor (False)
    scale_pos_weight : float, optional
        Weight to balance positive class (for XGBoost classifiers)

    Returns:
    --------
    model : sklearn/xgboost model instance
        Instantiated model ready for training
    """
    model_class = model_spec.model_class
    config = model_spec.config

    # Get parameters from config
    params = config.to_dict()
    
    # 2.A: Handle Class Imbalance for XGBoost
    if is_classifier and "XGBClassifier" in str(model_class) and scale_pos_weight is not None:
        # Only apply if not explicitly set in config
        if 'scale_pos_weight' not in params or params['scale_pos_weight'] is None:
            params['scale_pos_weight'] = scale_pos_weight

    # Instantiate the model directly with its parameters
    return model_class(**params)


def train_ensemble(
    model_specs: List[ModelSpec],
    X: pd.DataFrame,
    y: pd.Series,
    is_classifier: bool = True,
    use_stacking: bool = False,
    groups: np.ndarray = None
) -> List[Tuple[Any, float]]:
    """
    Train an ensemble of models and return list of (model, weight) tuples.

    Parameters:
    -----------
    model_specs : List[ModelSpec]
        List of model specifications
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    is_classifier : bool
        Whether training classifiers or regressors
    use_stacking : bool
        If True, use Stacking instead of weighted averaging (Not fully implemented in this function return type, 
        but prepared for future extension. For now, we stick to weighted averaging but improve individual models)
    groups : np.ndarray, optional
        Group labels for cross-validation (e.g., auction_month)

    Returns:
    --------
    ensemble : List[Tuple[model, weight]]
        Trained models with their weights
    """
    ensemble = []
    
    # Calculate scale_pos_weight for this batch
    scale_pos_weight = 1.0
    if is_classifier:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0:
            scale_pos_weight = n_neg / n_pos
    
    for spec in model_specs:
        model = create_model(spec, is_classifier=is_classifier, scale_pos_weight=scale_pos_weight)
        
        # Pass groups to fit if it's a StackingModel
        if isinstance(model, StackingModel):
            model.fit(X, y, groups=groups)
        else:
            model.fit(X, y)
            
        ensemble.append((model, spec.weight))
        
    return ensemble


def predict_ensemble(
    ensemble: List[Tuple[Any, float]],
    X: pd.DataFrame,
    predict_proba: bool = False
) -> np.ndarray:
    """
    Make predictions using an ensemble of models with weighted averaging.

    Parameters:
    -----------
    ensemble : List[Tuple[model, weight]]
        Trained models with their weights
    X : pd.DataFrame
        Features for prediction
    predict_proba : bool
        If True, return probabilities (for classifiers)

    Returns:
    --------
    predictions : np.ndarray
        Weighted averaged predictions
    """
    if len(ensemble) == 0:
        raise ValueError("Empty ensemble")

    predictions = None
    for model, weight in ensemble:
        if predict_proba:
            # For classifiers, get probability of positive class
            pred = model.predict_proba(X)[:, 1]
        else:
            # For regressors or class predictions
            pred = model.predict(X)

        if predictions is None:
            predictions = weight * pred
        else:
            predictions += weight * pred

    return predictions


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    beta: float = 0.5
) -> Tuple[float, float]:
    """
    Find optimal classification threshold that maximizes F-beta score.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    thresholds : array-like, optional
        Threshold values to search
    beta : float
        Beta parameter for F-beta score

    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes F-beta score
    max_fbeta : float
        Maximum F-beta score achieved
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    fbeta_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fbeta = fbeta_score(y_true, y_pred, average='weighted', beta=beta)
        fbeta_scores.append(fbeta)

    optimal_idx = np.argmax(fbeta_scores)
    optimal_threshold = thresholds[optimal_idx]
    max_fbeta = fbeta_scores[optimal_idx]

    return optimal_threshold, max_fbeta


class ShadowPriceModels:
    """Trains and stores ensemble of classification and regression models."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        # Ensembles: Dict[branch_name, List[Tuple[model, weight]]]
        self.clf_ensembles: Dict[str, List[Tuple[Any, float]]] = {}
        self.reg_ensembles: Dict[str, List[Tuple[Any, float]]] = {}
        self.optimal_thresholds: Dict[str, float] = {}
        # Default ensembles: List[Tuple[model, weight]]
        self.clf_default_ensemble: List[Tuple[Any, float]] = []
        self.reg_default_ensemble: List[Tuple[Any, float]] = []
        self.optimal_threshold_default: float = 0.5

    def train_classifiers(
        self,
        train_data: pd.DataFrame,
        test_branches: Set[str],
        verbose: bool = True
    ) -> None:
        """
        Train branch-specific classifiers and default fallback classifier.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels
        test_branches : set
            Set of branch names in test set
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("\n[Training Classification Models + Threshold Optimization]")
            print("-" * 80)

        # Group training data by branch
        train_data_by_branch = {
            branch_name: branch_data
            for branch_name, branch_data in train_data.groupby('branch_name')
        }

        # Train default classifier on all data
        if verbose:
            print("Training default fallback classifier on all training data...")

        X_train_all = train_data[self.config.features.step1_features].copy()
        y_train_all = train_data['label'].copy()
        y_train_binary_all = (y_train_all > 0).astype(int)
        
        # Extract groups (auction_month) for CV
        groups_all = train_data['auction_month'].values if 'auction_month' in train_data.columns else None

        # Train default classifier ensemble
        self.clf_default_ensemble = train_ensemble(
            self.config.models.default_classifiers,
            X_train_all,
            y_train_binary_all,
            is_classifier=True,
            groups=groups_all
        )

        if verbose:
            n_models = len(self.clf_default_ensemble)
            model_names = [spec.model_class.__name__ for spec in self.config.models.default_classifiers]
            print(f"  ✓ Default ensemble trained ({n_models} models: {', '.join(model_names)})")
            print(f"    Total samples: {len(X_train_all):,}")

        # Optimize threshold for default classifier ensemble
        if verbose:
            print(f"  Optimizing threshold for default ensemble...")

        y_proba_train_all = predict_ensemble(self.clf_default_ensemble, X_train_all, predict_proba=True)
        thresholds = np.linspace(
            self.config.threshold.threshold_range_start,
            self.config.threshold.threshold_range_end,
            self.config.threshold.threshold_range_steps
        )
        self.optimal_threshold_default, max_f1 = find_optimal_threshold(
            y_train_binary_all, y_proba_train_all, thresholds, self.config.threshold.threshold_beta
        )

        if verbose:
            print(f"  ✓ Optimal threshold (default): {self.optimal_threshold_default:.3f} (F-beta={max_f1:.3f})")

        # Train branch-specific classifiers
        if verbose:
            print(f"\nTraining branch-specific classifiers + optimizing thresholds...")
            print(f"  Total branches to train: {len(train_data_by_branch):,}")

        trained_count = 0
        skipped_count = 0

        for branch_name, branch_data in train_data_by_branch.items():
            # Only train models for branches in test set
            if branch_name not in test_branches:
                skipped_count += 1
                continue

            # Skip branches with too few samples
            if len(branch_data) < self.config.training.min_samples_for_branch_model:
                skipped_count += 1
                continue

            # Prepare branch data
            X_branch = branch_data[self.config.features.step1_features].copy()
            y_branch_binary = (branch_data['label'] > 0).astype(int)
            groups_branch = branch_data['auction_month'].values if 'auction_month' in branch_data.columns else None

            # Train branch-specific ensemble
            try:
                clf_ensemble = train_ensemble(
                    self.config.models.branch_classifiers,
                    X_branch,
                    y_branch_binary,
                    is_classifier=True,
                    groups=groups_branch
                )
                self.clf_ensembles[branch_name] = clf_ensemble

                # Optimize threshold using ensemble predictions
                y_proba_train_branch = predict_ensemble(clf_ensemble, X_branch, predict_proba=True)
                optimal_threshold_branch, _ = find_optimal_threshold(
                    y_branch_binary, y_proba_train_branch, thresholds, self.config.threshold.threshold_beta
                )
                self.optimal_thresholds[branch_name] = optimal_threshold_branch

                trained_count += 1
            except:
                skipped_count += 1
                continue

            if verbose and trained_count % 500 == 0:
                print(f"    Progress: {trained_count:,} models trained...")

        if verbose:
            print(f"\n✓ Classification Training Complete")
            print(f"  Models trained: {trained_count:,}")
            print(f"  Optimal thresholds computed: {len(self.optimal_thresholds):,}")
            print(f"  Branches skipped: {skipped_count:,}")
            print(f"  Default optimal threshold: {self.optimal_threshold_default:.3f}")

    def train_regressors(
        self,
        train_data: pd.DataFrame,
        test_branches: Set[str],
        verbose: bool = True
    ) -> None:
        """
        Train branch-specific regressors and default fallback regressor.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels
        test_branches : set
            Set of branch names in test set
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("\n[Training Regression Models]")
            print("-" * 80)

        # Group training data by branch
        train_data_by_branch = {
            branch_name: branch_data
            for branch_name, branch_data in train_data.groupby('branch_name')
        }

        # Train default regressor on all binding data
        if verbose:
            print("Training default fallback regressor on all binding training data...")

        X_train_all_reg = train_data[self.config.features.step2_features].copy()
        y_train_all_reg = train_data['label'].copy()
        groups_all_reg = train_data['auction_month'].values if 'auction_month' in train_data.columns else None

        binding_mask_all = y_train_all_reg > 0
        X_train_binding_all = X_train_all_reg[binding_mask_all]
        y_train_binding_all = y_train_all_reg[binding_mask_all]
        
        if groups_all_reg is not None:
            groups_binding_all = groups_all_reg[binding_mask_all]
        else:
            groups_binding_all = None

        if len(X_train_binding_all) > 10:
            # Train default regressor ensemble
            self.reg_default_ensemble = train_ensemble(
                self.config.models.default_regressors,
                X_train_binding_all,
                y_train_binding_all,
                is_classifier=False,
                groups=groups_binding_all
            )

            if verbose:
                n_models = len(self.reg_default_ensemble)
                model_names = [spec.model_class.__name__ for spec in self.config.models.default_regressors]
                print(f"  ✓ Default ensemble trained ({n_models} models: {', '.join(model_names)})")
                print(f"    Binding samples: {len(X_train_binding_all):,}")
                print(f"    Features: {len(self.config.features.step2_features)} - {self.config.features.step2_features}")
        else:
            if verbose:
                print(f"  ⚠️  Insufficient binding samples for default regressor")

        # Train branch-specific regressors
        if verbose:
            print(f"\nTraining branch-specific regressors...")
            print(f"  Total branches to train: {len(train_data_by_branch):,}")

        trained_reg_count = 0
        skipped_reg_count = 0

        for branch_name, branch_data in train_data_by_branch.items():
            if branch_name not in test_branches:
                skipped_reg_count += 1
                continue

            # Extract binding samples
            X_branch = branch_data[self.config.features.step2_features].copy()
            y_branch = branch_data['label'].copy()
            groups_branch = branch_data['auction_month'].values if 'auction_month' in branch_data.columns else None

            binding_mask_branch = y_branch > 0
            X_branch_binding = X_branch[binding_mask_branch]
            y_branch_binding = y_branch[binding_mask_branch]
            
            if groups_branch is not None:
                groups_branch_binding = groups_branch[binding_mask_branch]
            else:
                groups_branch_binding = None

            # Skip if not enough binding samples
            if len(X_branch_binding) < self.config.training.min_binding_samples_for_regression:
                skipped_reg_count += 1
                continue

            # Train branch-specific regressor ensemble
            try:
                reg_ensemble = train_ensemble(
                    self.config.models.branch_regressors,
                    X_branch_binding,
                    y_branch_binding,
                    is_classifier=False,
                    groups=groups_branch_binding
                )
                self.reg_ensembles[branch_name] = reg_ensemble
                trained_reg_count += 1
            except:
                skipped_reg_count += 1
                continue

            if verbose and trained_reg_count % 500 == 0:
                print(f"    Progress: {trained_reg_count:,} regression models trained...")

        if verbose:
            print(f"\n✓ Regression Training Complete")
            print(f"  Models trained: {trained_reg_count:,}")
            print(f"  Branches skipped: {skipped_reg_count:,}")
            print(f"  Default regressor: {'Available' if len(self.reg_default_ensemble) > 0 else 'Not available'}")
