
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from selection import RecursiveFeatureSelection


class BinaryClassificationVisualizer:
    """
    class BinaryClassificationVisualizer for plotting feature importance for binary classifiers
    """
    allowable_metrics = frozenset(['auc', 'precision', 'recall', 'f1'])

    def __init__(self, rfs: RecursiveFeatureSelection):
        """
        Instantiate an object of class BinaryClassificationVisualizer.

        :param rfs: fitted RecursiveFeatureSelection object.
        """
        self.rfs = rfs
        self.scores = None
        self.scored = False

    def reset_scores(self) -> None:
        """
        Reset the scores attribute.

        :return: None
        """
        self.scores = {x: [] for x in BinaryClassificationVisualizer.allowable_metrics}
        self.scored = False

    def score(self, X: pd.DataFrame, y: ArrayLike, threshold: float = 0.5):
        """
        Calculate the AUC, precision, recall, and F1 metrics across all models.

        :param X: pd.DataFrame; the input features.

        :param y: Arraylike; the target variable.

        :param threshold: float; default is 0.5.

        :return: None
        """
        self.reset_scores()
        pred_probas = [x[:, 1] for x in self.rfs.predict_proba(X)]
        for pred_proba in pred_probas:
            pred = np.where(pred_proba >= threshold, 1, 0)
            self.scores['auc'].append(roc_auc_score(y, pred_proba))
            self.scores['precision'].append(precision_score(y, pred))
            self.scores['recall'].append(recall_score(y, pred))
            self.scores['f1'].append(f1_score(y, pred))

    def plot(self, metric: str, xtick_interval: int = 5, ytick_interval: float = 0.05, *args, **kwargs) -> Tuple:
        """
        Plot the performance metric across all models as less important features are remove.

        :param metric: str; 'auc', 'precision', 'recall', 'f1'

        :param xtick_interval: int; default is 5.  The interval to draw x-axis ticks.

        :param ytick_interval: float; default is 0.05.  The interval to draw y-axis ticks.

        :param args: arguments to pass to plt.plot().

        :param kwargs: keyword arguments to pass to plt.plot().

        :return: fig, ax
        """
        if not self.scored:
            raise ValueError('Instance has not yet been scored.  Call the score() method before calling plot()')

        if metric not in BinaryClassificationVisualizer.allowable_metrics:
            raise ValueError(f'metric must be one of the following: {BinaryClassificationVisualizer.allowable_metrics}')

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
