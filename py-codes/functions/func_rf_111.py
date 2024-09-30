import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from functions.rtools.embed import embed

def run_rf(Y, lag, n_jobs):
    Y2 = Y
    aux = embed(Y2, 2 + lag)
    y = aux[:, 0]
    X = aux[:, Y2.shape[1] * lag:]

    if lag == 1:
        X_out = aux[-1, :X.shape[1]] # 마지막 행을 제거해서 예측에 사용
    else:
        X_out = aux[:, Y2.shape[1] * (lag-1):]
        X_out = X_out[-1, :X.shape[1]]

    # RandomForestRegressor에서 n_jobs를 사용해 병렬 처리
    model = RandomForestRegressor(n_estimators=501, n_jobs=n_jobs, random_state = 42)
    # TODO: 501개의 RF tree를 사용한게 맞는지
    model.fit(X, y)

    # 예측
    pred = model.predict(X_out.reshape(1, -1))

    return {"model": model, "pred": pred}

def rf_rolling_window(Y, npred, indice=1, lag=1, n_jobs = 3):
    save_importance = []
    save_pred = np.full(npred, np.nan)

    indice-=1 # for python, align the start to 0

    for i in range(npred, 0, -1):
        Y_window = Y[(npred - i):(Y.shape[0] - i), :]
        result = run_rf(Y_window, lag, n_jobs=n_jobs)
        save_pred[npred - i] = result['pred']

        # 랜덤 포레스트에서 변수 중요도 추출
        save_importance.append(result['model'].feature_importances_)
        print(f"iteration {npred - i + 1}")

    real = Y[:, indice]

    # 그래프 그리기
    plt.plot(real, label='Real')
    plt.plot(np.concatenate((np.full(len(real) - npred, np.nan), save_pred.flatten())), color='red', label='Prediction')
    plt.title(f"RF {lag}-day ahead forecast", fontsize = 10, weight = "bold")
    plt.legend()
    plt.show()

    # RMSE 및 MAE 계산
    real_tail = real[-npred:]
    rmse = mean_squared_error(real_tail, save_pred, squared = False)
    mae = mean_absolute_error(real_tail, save_pred)
    errors = {"rmse": rmse, "mae": mae}

    return {"pred": save_pred, "errors": errors, "save_importance": save_importance}
