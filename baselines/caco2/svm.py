import os
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVR
from sklearn import metrics

def svr(X_train, y_train, X_test, y_test, C, epsilon):

    svr_regressor = SVR(kernel='rbf', C=C, epsilon=epsilon)

    svr_regressor.fit(X_train, y_train)

    y_pred = svr_regressor.predict(X_test)

    r2 = metrics.r2_score(y_test, y_pred)

    mse = metrics.mean_squared_error(y_test, y_pred)

    mae = metrics.mean_absolute_error(y_test, y_pred)

    return r2, mse, mae

X_train = torch.load('../../data/caco2_for_baselines/X_train_fg.pt')

y_train = pd.read_csv('../../data/caco2_for_baselines/caco2_train.csv')['y'].values

X_test = torch.load('../../data/caco2_for_baselines/X_test_fg.pt')

y_test = pd.read_csv('../../data/caco2_for_baselines/caco2_test.csv')['y'].values


LOG_FILE = 'svm_baselines_result.csv'

C_testues = [0.1, 1, 10, 100]

epsilon_testues = [0.01, 0.1, 0.5, 1]

for repeat in [0, 1, 2]:
    for C in C_testues:
        for epsilon in epsilon_testues:
            np.random.seed(repeat)
            y_train = y_train + 0.01 * np.random.randn(len(y_train))
            r2, mse, mae = svr(X_train, y_train, X_test, y_test, C, epsilon)
            results = pd.DataFrame({"model":[f'svr_{C}_{epsilon}'], "repeat":[repeat], "r2":[r2], "mse":[mse], "mae":[mae]})
            results.to_csv(LOG_FILE, mode='a', index=False, header=False if os.path.isfile(LOG_FILE) else True)

