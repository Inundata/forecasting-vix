import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from functions.rtools.embed import embed

def run_ar(Y, lag, model_type="fixed"):
    Y2 = Y
    aux = embed(Y2, lag) # TODO: lag 1인 경우, 57, 63번째 열에 np.nan존재 // 앞에서 np.nan 처리안해서 연계로 발생
    y = aux[:, 0]
    X = aux[:, Y2.shape[1] * lag:] # lag를 고려하고 있는 지점을 가져오는 것, 
                                    # 즉 lag이 5이면 t-5부분의 X를 가져오는 것

    if lag == 1:
        X_out = aux[-1, :X.shape[1]] # 마지막 행을 제거해서 예측에 사용
    else:
        X_out = aux[:, Y2.shape[1] * (lag-1):]
        X_out = X_out[-1, :X.shape[1]]

    if model_type == "fixed":
        model = sm.OLS(y, sm.add_constant(X), missing = "drop").fit() # TODO : missing drop이 의도한건지
        coef = model.params
    elif model_type == "bic":
        bb = np.inf
        ar_coef = None
        for i in range(1, X.shape[1] + 1):
            model = sm.OLS(y, sm.add_constant(X[:, :i])).fit()
            crit = model.bic
            if crit < bb:
                bb = crit
                ar_coef = model.params
        coef = np.zeros(X.shape[1] + 1)
        coef[:len(ar_coef)] = ar_coef

    pred = np.append(1, X_out) @ coef

    return {"model": model, "pred": pred, "coef": coef}


def har_rolling_window(Y, npred, model_name, indice=1, lag=1, model_type="fixed") :
    save_coef = np.full((npred, 18), np.nan)
    save_pred = np.full(npred, np.nan)

    indice-=1 # for python, align the start to 0

    for i in range(npred, 0, -1):
        Y_window = Y[(npred - i):(Y.shape[0] - i), :]
        fact = run_ar(Y_window, lag, model_type)

        save_pred[npred - i] = fact['pred']
        # print(f"iteration {npred - i + 1}")

    real = Y[:, indice]

    # 그래프 그리기
    plt.plot(real, label='Real')
    plt.plot(np.concatenate((np.full(len(real) - npred, np.nan), save_pred.flatten())), color='red', label='Prediction')

    if model_name == "harx":
        plt.title(f"HARX {lag}-day ahead forecast", fontsize = 10, weight = "bold")
    elif model_name == "har":
        plt.title(f"HAR {lag}-day ahead forecast", fontsize = 10, weight = "bold")
    plt.legend()
    plt.show()

    # RMSE 및 MAE 계산
    real_tail = real[-npred:]
    rmse = mean_squared_error(real_tail, save_pred, squared = False)
    mae = mean_absolute_error(real_tail, save_pred)
    errors = {"rmse": rmse, "mae": mae}

    if model_name == "harx":
        print(f"HARX: {lag}-ahead forecast is finished")
    elif  model_name == "har":
        print(f"HAR: {lag}-ahead forecast is finished")

    return {"pred": save_pred, "coef": save_coef, "errors": errors}
