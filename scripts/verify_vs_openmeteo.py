#!/usr/bin/env python
import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = os.getenv("MODEL_DIR", "weather_plus/models")


def eval_one(df_path, model_name, unit_note=""):
    df = pd.read_parquet(df_path)
    X = df[
        ["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs", "baseline_diff"]
    ].values
    y = df["target"].values
    bom = df["baseline_om"].values
    bifs = np.nan_to_num(df["baseline_ifs"].values, nan=bom)
    art = joblib.load(os.path.join(MODEL_DIR, model_name))
    yhat = art.predict(X)
    mae_om = mean_absolute_error(y, bom)
    rmse_om = mean_squared_error(y, bom, squared=False)
    mae_ifs = mean_absolute_error(y, bifs)
    rmse_ifs = mean_squared_error(y, bifs, squared=False)
    mae = mean_absolute_error(y, yhat)
    rmse = mean_squared_error(y, yhat, squared=False)
    print(
        f"{model_name:26s} | MAE ours={mae:7.3f}  OM={mae_om:7.3f}  IFS={mae_ifs:7.3f} | "
        f"RMSE ours={rmse:7.3f} OM={rmse_om:7.3f} IFS={rmse_ifs:7.3f} {unit_note}"
    )


def main():
    items = [
        ("data/train/t2m_training.parquet", "temperature_2m.joblib", "°C"),
        ("data/train/td2m_training.parquet", "dew_point_2m.joblib", "°C"),
        ("data/train/psfc_training.parquet", "surface_pressure.joblib", "hPa"),
        ("data/train/tp_training.parquet", "precipitation.joblib", "mm"),
        ("data/train/wspd100_training.parquet", "wind_speed_100m.joblib", "km/h"),
        ("data/train/wdir100_training.parquet", "wind_direction_100m.joblib", "deg"),
    ]
    for path, m, unit in items:
        if os.path.exists(path) and os.path.exists(os.path.join(MODEL_DIR, m)):
            eval_one(path, m, unit_note=f"[{unit}]")
        else:
            print("skip", m)


if __name__ == "__main__":
    main()
