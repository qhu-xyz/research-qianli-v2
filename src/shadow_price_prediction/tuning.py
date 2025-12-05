"""
Hyperparameter tuning module.
"""

from typing import Any

import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from .config import ModelSpec
from .models import create_model


class HyperparameterTuner:
    """
    Handles hyperparameter optimization for models.
    """

    def __init__(self, n_iter: int = 20, cv: int = 3, verbose: int = 1):
        self.n_iter = n_iter
        self.cv = cv
        self.verbose = verbose

    def tune_model(
        self,
        model_spec: ModelSpec,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        param_distributions: dict[str, Any],
        is_classifier: bool = True,
        scoring: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Tune a single model using RandomizedSearchCV.

        Parameters:
        -----------
        model_spec : ModelSpec
            Base model specification
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        param_distributions : Dict[str, Any]
            Search space for parameters
        is_classifier : bool
            Whether tuning a classifier
        scoring : str
            Scoring metric (e.g., 'f1', 'neg_mean_squared_error')

        Returns:
        --------
        best_model : Any
            Tuned model instance
        best_params : Dict
            Best parameters found
        """
        # Create base model
        base_model = create_model(model_spec, is_classifier=is_classifier)

        # Default scoring
        if scoring is None:
            if is_classifier:
                # Custom F2 scorer for recall preference
                scoring = make_scorer(fbeta_score, beta=2)
            else:
                scoring = "neg_mean_squared_error"

        # TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=self.cv)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring=scoring,
            cv=tscv,
            verbose=self.verbose,
            n_jobs=1,  # Disable internal parallelism to avoid conflicts
            random_state=42,
        )

        if self.verbose:
            print(f"Tuning {model_spec.model_class.__name__}...")

        search.fit(X, y)

        if self.verbose:
            print(f"Best params: {search.best_params_}")
            print(f"Best score: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_

    def tune_ensemble(
        self,
        model_specs: list[ModelSpec],
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        param_grids: dict[str, dict[str, Any]],
        is_classifier: bool = True,
    ) -> list[tuple[Any, float]]:
        """
        Tune all models in an ensemble specification.
        """
        tuned_ensemble = []

        for spec in model_specs:
            model_name = spec.model_class.__name__

            if model_name in param_grids:
                # Tune this model
                best_model, _ = self.tune_model(spec, X, y, param_grids[model_name], is_classifier=is_classifier)
                tuned_ensemble.append((best_model, spec.weight))
            else:
                # No tuning grid, just train default
                if self.verbose:
                    print(f"No tuning grid for {model_name}, using default.")
                model = create_model(spec, is_classifier=is_classifier)
                model.fit(X, y)
                tuned_ensemble.append((model, spec.weight))

        return tuned_ensemble
