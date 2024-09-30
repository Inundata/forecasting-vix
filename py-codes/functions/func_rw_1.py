def rw_rolling_window(Y, npred, indice=1, lag=1):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    import matplotlib.pyplot as plt

    # indice: The target Y
    indice-=1 # for python, align the start to 0

    YY = np.array(Y[:, indice]) 

    save_pred = np.full(npred, np.nan)
    
    for i in range(npred, 0, -1):
        pred = YY[(len(YY) - i - lag)]
        save_pred[(npred - i)] = pred
        # print(f"iteration {npred - i + 1}")
    
    real = Y[:, indice]
    
    # 그래프 그리기
    plt.plot(real, label='Real')
    plt.plot(np.concatenate((np.full(len(real) - npred, np.nan), save_pred)), color='red', label='Prediction')
    plt.title(f"Random walk {lag}-day head forecast", fontsize = 10, weight = "bold")
    plt.legend()
    plt.show()
    
    # RMSE 및 MAE 계산
    real_tail = real[-npred:]
    rmse = mean_squared_error(real_tail, save_pred, squared = False)
    mae = mean_absolute_error(real_tail, save_pred)

    errors = {"rmse": rmse, "mae": mae}

    print(f"random walk: {lag}-ahead forecast is finished")
    
    return {"pred": save_pred, "errors": errors}