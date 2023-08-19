
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
    def get_importance(self, estimator, features: List[str]) -> pd.DataFrame:
        ...


class DefaultImportance(FeatureImportance):
    """
    class DefaultImportance which uses the feature_importances_ attribute after an estimator has been fitted
    """
    def get_importance(self, estimator: Estimator, features: List[str]) -> pd.DataFrame:
        """
        Get the feature importance.

        :param estimator: a fitted model object.

        :param features: list of str; the input features.

        :return: pd.DataFrame
        """
        return _sort_importance(features, estimator.feature_importances_)


class PermutationImportance(FeatureImportance):
    """
    class PermutationImportance for calculating feature importance using sklearn.inspection.permutation_importance.
    """
    def __init__(self, X: pd.DataFrame, y: ArrayLike, **kwargs):
        """
        Instantiate an object of class PermutationImportance

        :param X: pd.DataFrame; the input features

        :param y: ArrayLike; the target feature

        :param kwargs: keyword arguments to pass to sklearn.inspection.permutation_importance
        """
        self.X = X
        self.y = y
        self.kwargs = kwargs

    def get_importance(self, estimator: Estimator, features: List[str]) -> pd.DataFrame:
        """
        Get the feature importance.

        :param estimator: a fitted model object.

        :param features: list of str; the input features.

        :return: pd.DataFrame
        """

        values = permutation_importance(estimator, self.X[features], self.y, **self.kwargs).importances_mean
        return _sort_importance(features, values)
