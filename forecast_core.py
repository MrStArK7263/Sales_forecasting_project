"""
forecast_core.py
================
Single source of truth for the data preparation and feature engineering used by
the notebook (training), the FastAPI backend (serving) and the Streamlit app.
Keeping this logic in ONE file prevents "training/serving skew" (the model being
fed different features in production than it saw during training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Data loading: order lines -> one row per calendar day
# --------------------------------------------------------------------------- #
def load_daily_sales(csv_path: str) -> pd.Series:
    """Aggregate order-line data to DAILY total sales.

    * OrderDate is converted to a plain calendar date (time-zone stripped).
    * Sales of all order lines on the same day are summed.
    * Days with no orders are inserted with Sales = 0 (a complete daily
      calendar is required for lag / rolling features and for a daily forecast).
    """
    df = pd.read_csv(csv_path, usecols=["OrderDate", "Sales"])
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], utc=True).dt.tz_localize(None).dt.normalize()
    daily = df.groupby("OrderDate")["Sales"].sum().sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily.name = "sales"
    return daily.astype(float)


# --------------------------------------------------------------------------- #
# 2. Calendar features (always known in advance for any future date)
# --------------------------------------------------------------------------- #
CALENDAR_FEATURES = [
    "t", "day_of_week", "is_weekend", "day_of_month", "week_of_year", "month",
    "quarter", "day_of_year", "is_month_start", "is_month_end", "is_quarter_end",
    "is_december", "year_sin", "year_cos", "year_sin2", "year_cos2",
    "week_sin", "week_cos", "month_sin", "month_cos",
]


def calendar_features(dates, origin) -> pd.DataFrame:
    """Features that depend only on the date itself.  `origin` = first date of the
    training data, so `t` (trend counter) means the same thing everywhere."""
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=d)
    out["t"] = (d - pd.Timestamp(origin)).days.astype(float)
    out["day_of_week"] = d.dayofweek
    out["is_weekend"] = (d.dayofweek >= 5).astype(int)
    out["day_of_month"] = d.day
    out["week_of_year"] = d.isocalendar().week.to_numpy().astype(int)
    out["month"] = d.month
    out["quarter"] = d.quarter
    out["day_of_year"] = d.dayofyear
    out["is_month_start"] = d.is_month_start.astype(int)
    out["is_month_end"] = d.is_month_end.astype(int)
    out["is_quarter_end"] = d.is_quarter_end.astype(int)
    out["is_december"] = (d.month == 12).astype(int)
    doy = np.asarray(d.dayofyear)
    out["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["year_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    out["year_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    dow = np.asarray(d.dayofweek)
    out["week_sin"] = np.sin(2 * np.pi * dow / 7)
    out["week_cos"] = np.cos(2 * np.pi * dow / 7)
    mon = np.asarray(d.month)
    out["month_sin"] = np.sin(2 * np.pi * mon / 12)
    out["month_cos"] = np.cos(2 * np.pi * mon / 12)
    return out[CALENDAR_FEATURES]


# --------------------------------------------------------------------------- #
# 3. Lag / rolling features (use ONLY information from before the day)
# --------------------------------------------------------------------------- #
LAG_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_90",
    "roll_std_28", "roll_nonzero_rate_28", "days_since_last_sale",
]
MAX_LOOKBACK = 120  # rows of history needed to compute every lag feature


def lag_features(series: pd.Series) -> pd.DataFrame:
    """Lag/rolling features for every date in `series`.
    Every value is shifted by one day, so row t only sees days <= t-1 (no leakage)."""
    s = series.astype(float)
    prev = s.shift(1)
    out = pd.DataFrame(index=s.index)
    for k in (1, 7, 14, 28):
        out[f"lag_{k}"] = s.shift(k)
    for w in (7, 14, 28, 90):
        out[f"roll_mean_{w}"] = prev.rolling(w, min_periods=1).mean()
    out["roll_std_28"] = prev.rolling(28, min_periods=2).std()
    out["roll_nonzero_rate_28"] = (prev > 0).astype(float).rolling(28, min_periods=1).mean()
    # days since the last day that had any sales (capped at 90)
    had_sale = (prev > 0).to_numpy()
    pos = np.arange(len(s), dtype=float)
    last_sale_pos = pd.Series(np.where(had_sale, pos, np.nan), index=s.index).ffill().to_numpy()
    out["days_since_last_sale"] = np.clip(np.nan_to_num(pos - last_sale_pos, nan=90.0), 0, 90)
    return out[LAG_FEATURES]


# --------------------------------------------------------------------------- #
# 4. Training frame
# --------------------------------------------------------------------------- #
def build_frame(daily: pd.Series, use_lags: bool, origin=None) -> pd.DataFrame:
    origin = daily.index.min() if origin is None else origin
    X = calendar_features(daily.index, origin)
    if use_lags:
        X = pd.concat([X, lag_features(daily)], axis=1)
        X = X.iloc[28:]                      # drop warm-up rows that lack a 28-day lag
    return X


def feature_names(use_lags: bool) -> list:
    return CALENDAR_FEATURES + (LAG_FEATURES if use_lags else [])


# --------------------------------------------------------------------------- #
# 5. Forecasting
# --------------------------------------------------------------------------- #
def forecast(model, history: pd.Series, horizon: int, use_lags: bool,
             cols=None, origin=None, clip_at_zero: bool = True, max_value=None) -> pd.Series:
    """Forecast `horizon` days after the last date of `history`.

    * calendar-only models: predict all future dates in one shot.
    * lag models: RECURSIVE forecasting - predict day 1, append the prediction to
      the history, rebuild the lag features, predict day 2, ... until `horizon`.
    """
    origin = history.index.min() if origin is None else origin
    cols = list(cols) if cols is not None else feature_names(use_lags)
    future = pd.date_range(history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    cal = calendar_features(future, origin)

    if not use_lags:
        preds = np.asarray(model.predict(cal[cols]), dtype=float).ravel()
    else:
        buf = history.iloc[-MAX_LOOKBACK:].copy()
        preds = []
        for day in future:
            buf.loc[day] = np.nan                         # placeholder for the day to predict
            lf = lag_features(buf).iloc[[-1]]
            lf.index = [day]
            row = pd.concat([cal.loc[[day]], lf], axis=1)[cols]
            p = float(np.asarray(model.predict(row)).ravel()[0])
            if clip_at_zero:
                p = max(p, 0.0)
            if max_value is not None:
                p = min(p, float(max_value))
            buf.loc[day] = p
            preds.append(p)
        preds = np.array(preds)
    if clip_at_zero:
        preds = np.clip(preds, 0.0, None)
    if max_value is not None:
        preds = np.clip(preds, None, float(max_value))   # sanity guard against runaway extrapolation
    return pd.Series(preds, index=future, name="forecast")
