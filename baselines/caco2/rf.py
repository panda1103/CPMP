import os
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def rf(X_train, y_train, X_test, y_test, repeat):

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=repeat)

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    r2 = metrics.r2_score(y_test, y_pred)

    mse = metrics.mean_squared_error(y_test, y_pred)

    mae = metrics.mean_absolute_error(y_test, y_pred)

    return r2, mse, mae

X_train = torch.load('../../data/caco2_for_baselines/X_train_fg.pt')

y_train = pd.read_csv('../../data/caco2_for_baselines/caco2_train.csv')['y'].values

X_test = torch.load('../../data/caco2_for_baselines/X_test_fg.pt')

y_test = pd.read_csv('../../data/caco2_for_baselines/caco2_test.csv')['y'].values

LOG_FILE = 'rf_baselines_result.csv'

for repeat in [0, 1, 2, 3, 4]:
    r2, mse, mae = rf(X_train, y_train, X_test, y_test, repeat)
    results = pd.DataFrame({"model":['rf'], "repeat":[repeat], "r2":[r2], "mse":[mse], "mae":[mae]})
    results.to_csv(LOG_FILE, mode='a', index=False, header=False if os.path.isfile(LOG_FILE) else True)

