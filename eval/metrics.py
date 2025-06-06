import numpy as np

EPS = 1e-6

def _safe_div(a, b):
    return a / (b + EPS)

def calc_basic(pred, true):
    diff  = pred - true
    mse   = np.mean(diff**2)
    rmse  = np.sqrt(mse)
    mae   = np.mean(np.abs(diff))
    mape  = np.mean(_safe_div(np.abs(diff), np.abs(true))) * 100
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((true - true.mean())**2)
    r2    = 1.0 - ss_res / (ss_tot + EPS)
    return dict(MSE=mse, RMSE=rmse, MAE=mae, MAPE=mape, R2=r2)