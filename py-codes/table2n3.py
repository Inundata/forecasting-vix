import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager, rcParams
font_name = font_manager.FontProperties(fname = "c:/windows/fonts/malgun.ttf").get_name()
rc('font', family = font_name)
rcParams["axes.unicode_minus"] = False

from sklearn.metrics import mean_squared_error, mean_absolute_error

from boruta import BorutaPy

path = r"E:\OneDrive\Ph.D\98. Paper\1. vix-forecast\forecast-vix"

# custom
from functions.func_rw_1 import rw_rolling_window
from functions.func_ar_111 import har_rolling_window
from functions.func_ar_112 import ar_rolling_window
from functions.func_rf_111 import rf_rolling_window

# Line 7~16
df = pd.read_excel(f"{path}/data/dataset_HARX(14).xlsx")
# df.dropna(inplace = True) 
data = df.iloc[:, 1:].to_numpy()
data = data[66:, :]

date = df.loc[66:, "Code"]
cols_info = df.iloc[:, 1:].columns

npred = 3240

#TODO: SP_66day,Oil_66day np.nan이 존재함. 
# np.where(pd.isna(data))

Y1 = data[:5740,:]

# # Line 18 ~21
rw1 = rw_rolling_window(Y1, npred, 1, 1)
rw5 = rw_rolling_window(Y1, npred, 1, 5)
rw10 = rw_rolling_window(Y1, npred, 1, 10)
rw22 = rw_rolling_window(Y1, npred, 1, 22)

# Line 26 ~ 31
harx1 = har_rolling_window(Y1, npred, "harx", 1, 1, model_type = "fixed") # y_t = y_{t-1} + X_{t-1}
harx5 = har_rolling_window(Y1, npred, "harx", 1, 5, model_type = "fixed") # y_t = y_{t-5} + X_{t-5}
harx10 = har_rolling_window(Y1, npred, "harx", 1, 10, model_type = "fixed") # y_t = y_{t-10} + X_{t-10}
harx22 = har_rolling_window(Y1, npred, "harx", 1, 22, model_type = "fixed") # y_t = y_{t-10} + X_{t-10}

# Line 34 ~ 39
Y2 = Y1[:, :5] 
# cols info
# ['lnVIX', 'HAR_5day', 'HAR_10day', 'HAR_22day', 'HAR_66day']

har1 = har_rolling_window(Y2, npred, "har", 1, 1, model_type = "fixed")
har5 = har_rolling_window(Y2, npred, "har", 1, 5, model_type = "fixed")
har10 = har_rolling_window(Y2, npred, "har", 1, 10, model_type = "fixed")
har22 = har_rolling_window(Y2, npred, "har", 1, 22, model_type = "fixed")

# Line 46 ~ 51

Y3 = np.c_[Y1[:, :1], Y1[:, 5:19]] 
# col info 
# ['lnVIX', 'SP_1day', 'SP_5day', 'SP_10day', 'SP_22day', 'SP_66day',
# 'S&PCOMP(MV)', 'Oil_1day', 'Oil_5day', 'Oil_10day', 'Oil_22day',
# 'Oil_66day', 'T10Y3M', 'Credit Spread', 'USG6WIC(TW)']

arx1 = ar_rolling_window(Y3, npred, 1, 1, model_type = "fixed") # y_t = y_{t-1} + y_{t-2} + X_{t-1} + X_{t-2} 
arx5 = ar_rolling_window(Y3, npred, 1, 5, model_type = "fixed") # y_t = y_{t-5} + y_{t-4} + X_{t-5} + X_{t-4} 
arx10 = ar_rolling_window(Y3, npred, 1, 10, model_type = "fixed") # y_t = y_{t-10} + y_{t-9} + X_{t-10} + X_{t-9}
arx22 = ar_rolling_window(Y3, npred, 1, 22, model_type = "fixed") # y_t = y_{t-22} + y_{t-21} + X_{t-22} + X_{t-21}

# Line 45 ~ 61
# Random Forest(14)
n_jobs = 4
rf1_14 = rf_rolling_window(Y3, npred, 1, 1, n_jobs = n_jobs) # y_t = f(X_{t-1})

import pickle

# 객체를 pickle로 저장
with open(f'{path}/rf1_14.pkl', 'wb') as f:
    pickle.dump(rf1_14, f)