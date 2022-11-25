import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, fetch_covtype
import autosklearn.regression
import autosklearn.classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import dabl
import numpy as np
import pandas as pd
from flask import jsonify

def get_data_type(df_col):
  if not isinstance(df_col, pd.DataFrame):
    df_col = pd.DataFrame(df_col)
  dtypes = dabl.detect_types(df_col)
  type_index = np.argmax(dtypes)
  dtype = dtypes.columns[type_index]
  return dtype

def get_columns_with_type(df, type):
  ''' type = categorical|continuous '''
  t = dabl.detect_types(df)
  columns = list(t.loc[:, 'categorical'][lambda v: v == True].index)
  return columns

def preprocess(df, target_col):
  # Shuffle the dataset
  df = df.sample(frac=1, random_state=0)
  
  # Reset index
  df.reset_index(drop=True, inplace=True)

  # Cut out X and y
  y = df.pop(target_col).to_numpy()
  X = df

  # TODO:
  #   - Detect and normalize nan values
  #   - Replace nan values with interpolated values
  #   - One-hot-encoding of categorical features
  #   - Feature scaling/normalization

  # scaler = MinMaxScaler(feature_range=(0, 1))
  scaler = StandardScaler()
  rescaledX = scaler.fit_transform(X)

  return df, rescaledX, y

def split(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  return X_train, X_test, y_train, y_test

def get_data(name = 'california_housing', preview = False):
  df = None
  target = None

  if name == 'california_housing':
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame
    target = ds.target.name
  elif name == 'fetch_covtype':
    ds = fetch_covtype(as_frame=True)
    df = ds.frame
    target = ds.target.name
  
  if target and type(preview) is int:
    df = df[:preview]

  return df, target

def get_best_model(X, y):
  dtype = get_data_type(y)
  dtype_map = {
    'continuous': 'regression',
    'categorical': 'classification',
    'low_card_int': 'classification',
  }
  try:
    target_type = dtype_map[dtype]
  except KeyError:
    print('Cannot predict target since target column is neither continuous nor categorical.', target_type)
    return None

  automodel = None
  print(f'Detected target as a task for {target_type}')
  per_run_time_limit = 10
  if target_type == 'regression':
    automodel = autosklearn.regression.AutoSklearnRegressor(
      memory_limit=1000000,
      ensemble_kwargs = { 'ensemble_size': 1 },
      n_jobs=8,
      time_left_for_this_task=4 * per_run_time_limit,
      per_run_time_limit=per_run_time_limit
    )
  elif target_type == 'classification':
    automodel = autosklearn.classification.AutoSklearnClassifier(
      memory_limit=1000000,
      ensemble_kwargs = { 'ensemble_size': 1 },
      n_jobs=8,
      time_left_for_this_task=4 * per_run_time_limit,
      per_run_time_limit=per_run_time_limit
      # include = { 'classifier': ['random_forest'] },\
    )
  
  automodel.fit(X, y)
  return automodel


def predict_dataset(dataset_name):
  df, target_col = get_data(dataset_name)
  df, X, y = preprocess(df, target_col)
  X_train, X_test, y_train, y_test = split(X, y)
  automodel = get_best_model(X_train, y_train)
  predictions = automodel.predict(X_test, y_test)
  error_mape = mean_absolute_percentage_error(y_test, predictions)
  error_mae = mean_absolute_error(y_test, predictions)
  models_board = automodel.leaderboard()
  df_pred_test = pd.DataFrame({
    'actual': y_test,
    'predicted': predictions,
  })
  df_pred_test = df_pred_test[:20]
  return jsonify({
    'models_board': models_board.to_dict(),
    'predictions': df_pred_test.to_dict(),
    'error_mape': error_mape,
    'error_mae': error_mae,
  })




if __name__ == '__main__':
  # TODO: Derive from selection in UI
  df, target_col = get_data()

  # TODO: Display as a table in UI

  df, X, y = preprocess(df, target_col)
  X_train, X_test, y_train, y_test = split(X, y)
  automodel = get_best_model(X_train, y_train)
  predictions = automodel.predict(X_test, y_test)

  error_mape = mean_absolute_percentage_error(y_test, predictions)
  error_mae = mean_absolute_error(y_test, predictions)
  print(automodel.leaderboard())
  print(f'error_mape {error_mape}')
  print(f'error_mae {error_mae}')
  df_pred_test = pd.DataFrame({
    'actual': y_test,
    'predicted': predictions,
  })
  print(df_pred_test.head())