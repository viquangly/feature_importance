
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from selection import RecursiveFeatureSelection, FeatureSelectionVisualizer
import importance as imp


def make_train_test_split(
        ncols: int, train_size: int, test_size: int, random_state: int = 1234
):
    rng = np.random.default_rng(random_state)
    nrows = train_size + test_size

    X = pd.DataFrame(rng.normal(size=(nrows, ncols)))
    X.columns = [f'x{i}' for i in range(ncols)]
    y = pd.Series(rng.uniform(0, 1, size=nrows))
    y = (y >= .5).astype(int)

    X_train = X.iloc[:train_size]
    y_train = y[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = make_train_test_split(100, 1000, 1000)
rf = RandomForestClassifier()

rfs = RecursiveFeatureSelection(rf, 5)
rfs.fit(X_train, y_train)

rfs_vis = FeatureSelectionVisualizer(rfs)
rfs_vis.score(X_test, y_test)
rfs_vis.plot('f1')

rfs_pi = RecursiveFeatureSelection(rf, 5, importance_calculator=imp.PermutationImportance())
rfs_pi.fit(X_train, y_train)
