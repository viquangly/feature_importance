
from abc import ABC, abstractmethod
from typing import List

from numpy.typing import ArrayLike
import pandas as pd
from sklearn.inspection import permutation_importance

from _typing import Estimator


def _sort_importance(features: List[str], values: ArrayLike, ascending: bool = False) -> pd.DataFrame:
    """
    Helper function to sort feature importance.

    :param features: list of str representing the feature names.

    :param values: arraylike; the feature importances

    :param ascending: bool; default is False.  The ascending argument to pass to pd.DataFrame.sort_values()

    :return: pd.DataFrame; the feature importance in order
    """
    return pd.DataFrame({
        'feature': features,
        'importance': values
    }).sort_values('importance', ascending=ascending).reset_index(drop=True)


class FeatureImportance(ABC):
    """
    Base class FeatureImportance for calculating feature importance
    """
    @abstractmethod
    def get_importance(self, estimator, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
        ...


class DefaultImportance(FeatureImportance):
    """
    class DefaultImportance which uses the feature_importances_ attribute after an estimator has been fitted
    """
    def get_importance(self, estimator: Estimator, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
        """
        Get the feature importance.

        :param estimator: a fitted model object.

        :param X: pd.DataFrame; the input features.

        :param y: Arraylike; the target variable

        :return: pd.DataFrame
        """
        return _sort_importance(list(X.columns), estimator.feature_importances_)


class PermutationImportance(FeatureImportance):
    """
    class PermutationImportance for calculating feature importance using sklearn.inspection.permutation_importance.
    """
    def __init__(self, **kwargs):
        """
        Instantiate an object of class PermutationImportance

        :param kwargs: keyword arguments to pass to sklearn.inspection.permutation_importance
        """
        self.kwargs = kwargs

    def get_importance(self, estimator, X: pd.DataFrame, y: ArrayLike) -> pd.DataFrame:
        """
        Get the feature importance.

        :param estimator: a fitted model object.

        :param X: pd.DataFrame; the input features.

        :param y: Arraylike; the target variable

        :return: pd.DataFrame
        """
        features = list(X.columns)
        values = permutation_importance(estimator, X, y, **self.kwargs).importances_mean
        return _sort_importance(features, values)
