
from typing import Optional, Union
from sklearn.linear_model import LogisticRegression
import importance as imp

Numeric = Union[float, int]


def _check_n_remove(value: Numeric) -> Numeric:
    if isinstance(value, int):
        if value < 1:
            raise ValueError('When n_remove is of type int, value must be >= 1')
    elif isinstance(value, float):
        if not (0 < value < 1):
            raise ValueError('When n_remove is of type float, value must be between 0 and 1 exclusive')
    else:
        raise TypeError('n_remove must be int or float')
    return value


def _check_min_features(value: Numeric) -> Numeric:
    if isinstance(value, int):
        if value < 1:
            raise ValueError('When min_features is of type int, value must be >= 1')
    elif isinstance(value, float):
        if not (0 < value < 1):
            raise ValueError('When min_features is of type float, value must be between 0 and 1 exclusive')
    else:
        raise TypeError('min_features must be int or float')
    return value


def _check_max_iter(value: Optional[int]) -> Numeric:
    if value is None:
        return float('inf')
    elif isinstance(value, int):
        if value < 1:
            raise ValueError('When max_iter is of type int, value must be >= 1')
    else:
        raise TypeError('max_iter must be None or int')

    return value


def _check_importance_calculator(estimator, value: Optional[imp.FeatureImportance]) -> imp.FeatureImportance:
    if value is None:
        value = imp.DefaultImportance()

    if not isinstance(value, imp.FeatureImportance):
        raise TypeError('importance_calculator must be of type FeatureImportance')

    if isinstance(estimator, LogisticRegression) and isinstance(value, imp.DefaultImportance):
        raise ValueError('Cannot use DefaultImportance with LogisticRegression.  Use PermutationImportance instead.')

    return value
