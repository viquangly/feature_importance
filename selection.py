
from copy import deepcopy
from typing import Union, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import importance as imp
import arg_checks as ac


class RecursiveFeatureSelection:

    def __init__(
            self, model, n_remove: Union[int, float], min_features: int = 1, max_iter: Optional[int] = None,
            importance_calculator: Optional[imp.FeatureImportance] = None
    ):
        self.base_model = model
        self.n_remove = ac.check_n_remove(n_remove)
        self.min_features = ac.check_min_features(min_features)
        self.max_iter = ac.check_max_iter(max_iter)
        self.input_features = []
        self.models = []
        self.feature_importances = []
        self.importance_calculator = ac.check_importance_calculator(importance_calculator)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index: int):
        return self.models[index], self.input_features[index], self.feature_importances[index]

    def _copy(self):
        return deepcopy(self.base_model)

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
        self.models = []
        self.feature_importances = []

        nrows, ncols = X.shape
        min_features = int(self.min_features * ncols) if isinstance(self.min_features, float) else self.min_features

        i = 0

        while True:
            model = self._copy()
            model.fit(X, y, **kwargs)
            self.models.append(model)
            self.input_features.append(list(X.columns))
            self._get_importance(model, X, y)

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
        return [model.predict(X[features]) for features, model in zip(self.input_features, self.models)]

    def predict_proba(self, X: pd.DataFrame) -> List[ArrayLike]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be of type pd.DataFrame')
        return [model.predict_proba(X[features]) for features, model in zip(self.input_features, self.models)]


class FeatureSelectionVisualizer:

    allowable_metrics = frozenset(['auc', 'precision', 'recall', 'f1'])

    def __init__(self, rfs: RecursiveFeatureSelection):
        self.rfs = rfs
        self.scores = None

    def create_score_dict(self):
        self.scores = {x: [] for x in FeatureSelectionVisualizer.allowable_metrics}

    def score(self, X: pd.DataFrame, y: ArrayLike, threshold: float = 0.5):
        self.create_score_dict()
        pred_probas = [x[:, 1] for x in self.rfs.predict_proba(X)]
        for pred_proba in pred_probas:
            pred = np.where(pred_proba >= threshold, 1, 0)
            self.scores['auc'].append(roc_auc_score(y, pred_proba))
            self.scores['precision'].append(precision_score(y, pred))
            self.scores['recall'].append(recall_score(y, pred))
            self.scores['f1'].append(f1_score(y, pred))

    def plot(self, metric: str, xtick_interval: int = 5, ytick_interval: float = 0.05, *args, **kwargs) -> Tuple:
        if metric not in FeatureSelectionVisualizer.allowable_metrics:
            raise ValueError(f'metric must be one of the following: {FeatureSelectionVisualizer.allowable_metrics}')

        y = self.scores[metric]
        plt.figure()
        plt.plot(range(len(y)), y, *args, **kwargs)
        plt.xticks(range(0, len(y), xtick_interval), rotation=90)
        plt.xlabel('index')
        plt.yticks(np.arange(0, 1 + ytick_interval, ytick_interval))
        plt.ylabel(metric)
        plt.ylim((0, 1))
        plt.grid()
        return plt.gcf(), plt.gca()
