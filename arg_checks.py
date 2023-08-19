
from typing import Optional
from sklearn.linear_model import LogisticRegression
import importance as imp

from _typing import Numeric, Estimator


def _check_n_remove(value: Numeric) -> Numeric:
    """
    Helper function to check n_remove argument is acceptable.

    :param value: numeric; n_remove argument

    :return: numeric
    """
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
    """
    Helper function to check if min_features argument is acceptable.

    :param value: numeric; min_features argument

    :return: numeric
    """
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
    """
    Helper function to check if max_iter argument is acceptable.

    :param value: optional; int.  The max_iter argument.

    :return: numeric
    """
    if value is None:
        return float('inf')
    elif isinstance(value, int):
        if value < 1:
            raise ValueError('When max_iter is of type int, value must be >= 1')
    else:
        raise TypeError('max_iter must be None or int')

    return value


def _check_importance_calculator(
        estimator: Estimator, value: Optional[imp.FeatureImportance]
) -> imp.FeatureImportance:
    """
    Check if the importance_calculator argument is acceptable.

    :param estimator: a fitted model object.

    :param value: optional; FeatureImportance.  The importance_calculator argument.

    :return: FeatureImportance
    """
    if value is None:
        value = imp.DefaultImportance()

    if not isinstance(value, imp.FeatureImportance):
        raise TypeError('importance_calculator must be of type FeatureImportance')

    if isinstance(estimator, LogisticRegression) and isinstance(value, imp.DefaultImportance):
        raise ValueError('Cannot use DefaultImportance with LogisticRegression.  Use PermutationImportance instead.')

    return value
