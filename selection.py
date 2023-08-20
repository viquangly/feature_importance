
from collections import Counter
from copy import deepcopy
from datetime import datetime
import itertools
from typing import Optional, List, Tuple

from numpy.typing import ArrayLike
import pandas as pd

import importance as imp
import arg_checks as ac
from _typing import Estimator, Numeric


def timestamp() -> str:
    """
    Helper function to get the current timestamp.

    :return: str; the current timestamp.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class RecursiveFeatureSelection:
    """
    class RecursiveFeatureSelection to remove less important features and refit the model at each iteration.
    """
    def __init__(
            self, estimator: Estimator, n_remove: Numeric, min_features: Numeric = 1, max_iter: Optional[int] = None,
            importance_calculator: Optional[imp.FeatureImportance] = None, verbose: bool = True
    ):
        """
        Instantiate an object of class RecursiveFeatureSelection.

        :param estimator: estimator object.  The estimator should not be fitted.

        :param n_remove: numeric; the number of feature to remove at each iteration.
        If n_remove is an int, each iteration will remove the specified number of the least important features.
        If n_remove is a float, each iteration will remove the specified percentage of the least important features.
        For example, if n_remove is 0.2, then the bottom 20% of features are removed.  Note: n_remove is only activated
        after all zero importance features are removed.

        :param min_features: numeric; the number of features to keep in the model.  Default is 1.  If min_features is
        a float, then it is treated as a percentage.  For example, if min_features is 0.2, and there are 100 features
        to start with, the model will have at least 20 features.

        :param max_iter: optional; int.  The max number of iterations.  If None, object will keep removing features
        until it hits min_features threshold.

        :param importance_calculator: optional; FeatureImportance object.  If None, DefaultImportance object is used.

        :param verbose: bool; default is True.  If True, print timestamp of each iteration.
        """
        self.base_estimator = estimator
        self.n_remove = ac._check_n_remove(n_remove)
        self.min_features = ac._check_min_features(min_features)
        self.max_iter = ac._check_max_iter(max_iter)
        self.input_features = []
        self.estimators = []
        self.feature_importances = []
        self.importance_calculator = ac._check_importance_calculator(estimator, importance_calculator)
        self.verbose = verbose

    def __len__(self) -> int:
        """
        Get the number of estimators.

        :return: int
        """
        return len(self.estimators)

    def __getitem__(self, index: int) -> Tuple[Estimator, List[str], pd.DataFrame]:
        """
        Get the estimator at the specified index.

        :param index: int

        :return: 3-element tuple of the estimator, input features, and feature importance
        """
        return self.estimators[index], self.input_features[index], self.feature_importances[index]

    def _copy(self) -> Estimator:
        """
        Make a copy of the base estimator

        :return: estimator object
        """
        return deepcopy(self.base_estimator)

    def _stop_for_max_iter(self, curr_iter: int) -> bool:
        """
        Check if iterations should stop due to exceeding max_iter threshold.

        :param curr_iter: int; the current iteration

        :return: bool. If True, iteration should stop.
        """
        return curr_iter >= self.max_iter

    @staticmethod
    def _stop_for_min_features(min_features: int, n_features: int) -> bool:
        """
        Check to see if the number of features in current iteration is less than the min_features threshold.

        :param min_features: int; the threshold.

        :param n_features: int; the number of features in the current iteration.

        :return: bool. If True, iteration should stop.
        """
        return n_features < min_features

    def _stop_for_no_change(self, n_features: int) -> bool:
        """
        Check to see if the number of features in current iteration is the same as the number of features in the previous
        iteration.

        :param n_features: int; the number of feature in the current iteration.

        :return: bool. If True, iteration should stop.
        """
        return n_features >= self._get_previous_n_features()

    def _get_previous_n_features(self) -> int:
        """
        Get the number of features in the previous iteration.

        :return: int; the number of features in the previous iteration.
        """
        return len(self.input_features[-1])

    def _identify_features_to_keep(self, n_remove: int) -> List[str]:
        """
        Identify the features to keep for next iteration.

        :param n_remove: int; the number of features to remove.

        :return: list of str representing the features to keep
        """
        sorted_feature_importances = self.feature_importances[-1]
        zeros_removed = sorted_feature_importances[sorted_feature_importances['importance'] == 0]
        if not zeros_removed.empty:
            return sorted_feature_importances.loc[
                ~sorted_feature_importances['feature'].isin(zeros_removed['feature']),
                'feature'
            ].tolist()
        return sorted_feature_importances['feature'][:(-n_remove)].tolist()

    def _calculate_n_remove(self) -> int:
        """
        Helper method to calculate the number of feature to remove at each iteration.

        :return: int
        """
        if isinstance(self.n_remove, float):
            return int(self.n_remove * self._get_previous_n_features())
        return self.n_remove

    def _get_importance(self) -> None:
        """
        Get the feature importance of the current iteration's estimator.

        :return: None
        """
        estimator = self.estimators[-1]
        features = self.input_features[-1]
        self.feature_importances.append(
            self.importance_calculator._get_importance(estimator, features)
        )

    def fit(self, X: pd.DataFrame, y: ArrayLike, **kwargs) -> None:
        """
        Fit models iteratively - removing less important features with each iteration.

        :param X: pd.DataFrame; the input features.

        :param y: ArrayLike; the target feature.

        :param kwargs: keyword arguments to pass to estimator.fit()

        :return: None
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pd.DataFrame')

        self.input_features = []
        self.estimators = []
        self.feature_importances = []

        nrows, ncols = X.shape
        min_features = int(self.min_features * ncols) if isinstance(self.min_features, float) else self.min_features

        i = 0

        while True:
            estimator = self._copy()
            estimator.fit(X, y, **kwargs)
            self.estimators.append(estimator)
            self.input_features.append(list(X.columns))
            self._get_importance()

            if self.verbose:
                n_features = self._get_previous_n_features()
                print(f'Iteration {i} - Fitted {n_features} features - {timestamp()}')

            i += 1
            n_features_to_remove = self._calculate_n_remove()
            next_iter_features = self._identify_features_to_keep(n_features_to_remove)
            next_iter_n_features = len(next_iter_features)
            X = X[next_iter_features]

            if any([
                self._stop_for_min_features(min_features, next_iter_n_features),
                self._stop_for_max_iter(i),
                self._stop_for_no_change(next_iter_n_features)
            ]):
                break

    def predict(self, X: pd.DataFrame) -> List[ArrayLike]:
        """
        Calculate the predictions of each iteration's model.

        :param X: pd.DataFrame; the input features.

        :return: list of ArrayLike
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pd.DataFrame')
        return [estimator.predict(X[features]) for features, estimator in zip(self.input_features, self.estimators)]

    def predict_proba(self, X: pd.DataFrame) -> List[ArrayLike]:
        """
        Calculate the prediction probabilities of each iteration's model.

        :param X: pd.DataFrame; the input features.

        :return: list of ArrayLike
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pd.DataFrame')
        return [
            estimator.predict_proba(X[features]) for features, estimator in zip(self.input_features, self.estimators)
        ]

    def get_feature_last_index(self, grouped: bool = False, ascending: bool = True) -> pd.DataFrame:
        """
        Show the last index in which the feature appeared.

        :param grouped: bool; default is False.  If False, have one row per feature.  If True, have one row per index.

        :param ascending: bool; default is True.  If True, place features with lower indices (first to leave) at top
        of table.

        :return: pd.DataFrame
        """

        if not self.input_features:
            raise ValueError('object has not been fitted.  Call .fit() method first.')

        last_index = pd.DataFrame(
            Counter(itertools.chain(*self.input_features)).items(),
            columns=['feature', 'last_index']
        ).sort_values('feature')
        shift_for_0index = 1
        last_index['last_index'] -= shift_for_0index

        if grouped:
            last_index = last_index.groupby('last_index')['feature'].apply(lambda x: ', '.join(x)).reset_index()

        return last_index[['last_index', 'feature']].sort_values(
            ['last_index', 'feature'], ascending=[ascending, True]
        ).reset_index(drop=True)
