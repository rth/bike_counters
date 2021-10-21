import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = 'Bike count prediction'
_target_column_name = 'bike_count'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()

score_types = [
    rw.score_types.RelativeRMSE(name='relative_rmse', precision=3),
    rw.score_types.RMSE(name='rmse', precision=3),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    for train_idx, test_idx in cv.split(X):
        yield train_idx, test_idx


def _read_data(path, f_name):
    data = pd.read_parquet(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.parquet'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.parquet'
    return _read_data(path, f_name)
