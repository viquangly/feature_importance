
from copy import deepcopy
from datetime import datetime
from typing import Optional, List

from numpy.typing import ArrayLike
import pandas as pd

import importance as imp
import arg_checks as ac
from _typing import Estimator, Numeric


def timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class RecursiveFeatureSelection:

    def __init__(
            self, estimator: Estimator, n_remove: Numeric, min_features: Numeric = 1, max_iter: Optional[int] = None,
            importance_calculator: Optional[imp.FeatureImportance] = None, verbose: bool = True
    ):
        self.base_estimator = estimator
        self.n_remove = ac._check_n_remove(n_remove)
        self.min_features = ac._check_min_features(min_features)
        self.max_iter = ac._check_max_iter(max_iter)
        self.input_features = []
        self.estimators = []
        self.feature_importances = []
        self.importance_calculator = ac._check_importance_calculator(estimator, importance_calculator)
        self.verbose = verbose

    def __len__(self):
        return len(self.estimators)

    def __getitem__(self, index: int):
        return self.estimators[index], self.input_features[index], self.feature_importances[index]

    def _copy(self):
        return deepcopy(self.base_estimator)

    def _stop_for_max_iter(self, curr_iter: int) -> bool:
        return curr_iter >= self.max_iter

    @staticmethod
    def _stop_for_min_features(min_features: int, n_features: int) -> bool:
        return n_features < min_features

    def _stop_for_no_change(self, n_features) -> bool:
        return n_features >= self._get_previous_n_features()

    def _get_previous_n_features(self) -> int:
        return len(self.input_features[-1])

    def _remove_least_important_features(self, n_remove: int) -> List[str]:
        sorted_feature_importances = self.feature_importances[-1]
        zeros_removed = sorted_feature_importances[sorted_feature_importances['importance'] == 0]
        if not zeros_removed.empty:
            return sorted_feature_importances.loc[
                ~sorted_feature_importances['feature'].isin(zeros_removed['feature']),
                'feature'
            ].tolist()
        return sorted_feature_importances['feature'][:(-n_remove)].tolist()

    def _calculate_n_remove(self) -> int:
        if isinstance(self.n_remove, float):
            return int(self.n_remove * self._get_previous_n_features())
        return self.n_remove

    def _get_importance(self, estimator, X: pd.DataFrame, y: ArrayLike) -> None:
        self.feature_importances.append(
            self.importance_calculator.get_importance(estimator, X, y)
        )

    def fit(self, X: pd.DataFrame, y: ArrayLike, **kwargs) -> None:

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
            self._get_importance(estimator, X, y)

            if self.verbose:
                n_features = self._get_previous_n_features()
                print(f'Iteration {i} - Fitted {n_features} features - {timestamp()}')

            i += 1
            n_features_to_remove = self._calculate_n_remove()
            next_iter_features = self._remove_least_important_features(n_features_to_remove)
            next_iter_n_features = len(next_iter_features)
            X = X[next_iter_features]

            if any([
                self._stop_for_min_features(min_features, next_iter_n_features),
                self._stop_for_max_iter(i),
                self._stop_for_no_change(next_iter_n_features)
            ]):
                break

    def predict(self, X: pd.DataFrame) -> List[ArrayLike]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pd.DataFrame')
        return [estimator.predict(X[features]) for features, estimator in zip(self.input_features, self.estimators)]

    def predict_proba(self, X: pd.DataFrame) -> List[ArrayLike]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pd.DataFrame')
        return [estimator.predict_proba(X[features]) for features, estimator in zip(self.input_features, self.estimators)]
