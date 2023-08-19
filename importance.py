
from abc import ABC, abstractmethod
from typing import List

from numpy.typing import ArrayLike
import pandas as pd
from sklearn.inspection import permutation_importance


def _sort_importance(features: List[str], values: ArrayLike) -> pd.DataFrame:
    return pd.DataFrame({
        'feature': features,
        'importance': values
    }).sort_values('importance', ascending=False).reset_index(drop=True)


class FeatureImportance(ABC):

    @abstractmethod
    def get_importance(self, estimator, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
        ...


class DefaultImportance(FeatureImportance):

    def get_importance(self, estimator, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
        return _sort_importance(list(X.columns), estimator.feature_importances_)


class PermutationImportance(FeatureImportance):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_importance(self, estimator, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
        features = list(X.columns)
        values = permutation_importance(estimator, X, y, **self.kwargs).importances_mean
        return _sort_importance(features, values)
