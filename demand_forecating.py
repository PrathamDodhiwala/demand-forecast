"""
demand_forecast_app.py

Single-file Streamlit app for demand forecasting.
- Upload CSV with date and target (demand) columns, or use included sample.
- Feature engineering (lags, rolling stats, date parts).
- Train RandomForest (or XGBoost if installed).
- Forecast future periods and download results.

Run:
    streamlit run demand_forecast_app.py
"""

import io
import math
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import streamlit as st

# Try to import xgboost
try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Demand Forecasting App", layout="wide")


# ---------------------
# Utility functions
# ---------------------
def create_lag_features(df, date_col, target_col, lags=[1, 7, 14, 28], windows=[7, 14]):
    df_sorted = df.sort_values(date_col).copy()
    for lag in lags:
        df_sorted[f"lag_{lag}"] = df_sorted[target_col].shift(lag)
    for w in windows:
        df_sorted[f"rolling_mean_{w}"] = (
            df_sorted[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        )
        df_sorted[f"rolling_std_{w}"] = (
            df_sorted[target_col]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .std()
            .fillna(0)
        )
    return df_sorted


def create_date_features(df, date_col):
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["dayofyear"] = df[date_col].dt.dayofyear
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    return df


def prepare_features(df, date_col, target_col, lags, windows):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = create_lag_features(df, date_col, target_col, lags=lags, windows=windows)
    df = create_date_features(df, date_col)
    # Drop rows with NaNs produced by lags
    df = df.dropna().reset_index(drop=True)
    return df


def train_model(X_train, y_train, model_name="RandomForest", random_state=42):
    if model_name == "XGBoost" and XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=200, learning_rate=0.05, random_state=random_state, verbosity=0
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}


def make_future_dataframe(last_date, periods, freq):
    return pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=periods,
        freq=freq,
    )


def iterative_forecast(
    model, history_df, date_col, target_col, features_used, future_dates, lags, windows
):
    """
    Iteratively forecast future_dates (DatetimeIndex), using lag features created from prior predictions.
    history_df must contain the last rows of the original series (with date_col and target_col).
    """
    hist = history_df.sort_values(date_col).copy()
    preds = []
    df_work = hist.copy()
    for dt in future_dates:
        row = {}
        row[date_col] = dt
        # Create lag/rolling/date features based on current df_work
        # Append a placeholder row for dt with NaN target to compute features
        temp = df_work.append({date_col: dt, target_col: np.nan}, ignore_index=True)
        temp[date_col] = pd.to_datetime(temp[date_col])
        temp = create_lag_features(
            temp, date_col, target_col, lags=lags, windows=windows
        )
        temp = create_date_features(temp, date_col)
        # The last row corresponds to dt
        feat_row = temp.iloc[-1]
        X_row = feat_row[features_used].values.reshape(1, -1)
        yhat = model.predict(X_row)[0]
        preds.append({"date": dt, "prediction": float(yhat)})
        # append predicted value to df_work as actual to allow next iteration lags
        df_work = df_work.append({date_col: dt, target_col: yhat}, ignore_index=True)
    return pd.DataFrame(preds)


# ---------------------
# Streamlit UI
# ---------------------
st.title("📈 Demand Forecasting — Python + Streamlit")

st.markdown(
    """
Upload a time series CSV (date column + demand/target column).  
The app will construct lag & rolling features, train a model, and produce future forecasts.
"""
)

with st.sidebar:
    st.header("Options")
    sample_data = st.checkbox("Load sample data (Airline passengers-like)", value=True)
    date_col = st.text_input("Date column name", value="date")
    target_col = st.text_input("Target column name", value="demand")
    freq = st.selectbox(
        "Frequency (pandas offset alias)",
        options=["D", "W", "M"],
        index=0,
        help="D = daily, W = weekly, M = month-end. Used for future forecast spacing.",
    )
    test_size = st.number_input(
        "Test size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05
    )
    forecast_periods = st.number_input(
        "Forecast horizon (periods)", min_value=1, max_value=365, value=30
    )
    model_choice = st.selectbox(
        "Model", options=["RandomForest"] + (["XGBoost"] if XGB_AVAILABLE else [])
    )
    st.write("Feature params")
    lags_input = st.text_input("Lags (comma separated)", value="1,7,14,28")
    windows_input = st.text_input("Rolling windows (comma separated)", value="7,14")
    run_button = st.button("Train & Forecast")

# Load data section
if sample_data:
    # Create a sample dataset (synthetic seasonal daily demand)
    rng = pd.date_range("2020-01-01", periods=1000, freq="D")
    np.random.seed(42)
    seasonal = 10 + 5 * np.sin(2 * np.pi * rng.dayofyear / 365.25)
    trend = np.linspace(0, 5, len(rng))
    noise = np.random.normal(0, 1.2, size=len(rng))
    demand = (50 + seasonal + trend + noise).round(2)
    df = pd.DataFrame({date_col: rng, target_col: demand})
    st.success("Sample data loaded (synthetic daily series).")
    st.dataframe(df.tail(6))
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("File uploaded.")
            st.dataframe(df.head(6))
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV or toggle 'Load sample data'.")
        df = None


# Parse lags/windows
def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]


lags = parse_int_list(lags_input)
windows = parse_int_list(windows_input)

if df is not None and run_button:
    st.subheader("Preparing data & features")

    # Validate columns
    if date_col not in df.columns or target_col not in df.columns:
        st.error(
            f"Columns not found. Make sure your data has columns named '{date_col}' and '{target_col}'."
        )
        st.stop()

    # Convert date col
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Couldn't parse dates in column '{date_col}': {e}")
        st.stop()

    # Sort and aggregate if necessary (if duplicates in a period)
    df = df.sort_values(date_col).reset_index(drop=True)
    st.write("Data range:", df[date_col].min().date(), "→", df[date_col].max().date())
    st.write("Total rows:", len(df))

    # Prepare features (this will drop rows with NaNs from lagging)
    prepared = prepare_features(
        df[[date_col, target_col]], date_col, target_col, lags=lags, windows=windows
    )
    st.write("After feature engineering, rows:", len(prepared))

    # Train/Test split by time (keep chronological order)
    split_idx = int(len(prepared) * (1 - test_size))
    train_df = prepared.iloc[:split_idx].copy()
    test_df = prepared.iloc[split_idx:].copy()

    # Features for model: all except date & target
    features = [c for c in prepared.columns if c not in [date_col, target_col]]
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    st.write("Training rows:", len(X_train), "| Test rows:", len(X_test))
    st.write("Features used:", features)

    # Train model
    with st.spinner("Training model..."):
        model = train_model(X_train, y_train, model_name=model_choice)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    metrics = evaluate(y_test, y_pred_test)
    st.subheader("Model performance on test set")
    st.write(metrics)

    # Show actual vs predicted plot for the test period
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_df[date_col], y_test, label="actual")
    ax.plot(test_df[date_col], y_pred_test, label="predicted")
    ax.set_title("Actual vs Predicted (test set)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

    # Forecast future periods (iterative using previous preds for lag features)
    last_known_date = df[date_col].max()
    future_dates = make_future_dataframe(last_known_date, int(forecast_periods), freq)
    st.subheader("Forecast")
    st.write(
        f"Forecasting {len(future_dates)} future periods from {last_known_date.date()} with freq='{freq}'"
    )

    # Use last max(lags + windows) rows as history
    history_needed = max(lags + windows) + 5
    history_df = (
        df[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)
    )
    history_df_tail = history_df.tail(history_needed).reset_index(drop=True)

    future_df = iterative_forecast(
        model,
        history_df_tail,
        date_col,
        target_col,
        features,
        future_dates,
        lags,
        windows,
    )
    st.dataframe(future_df.head(20))

    # Combine with historical for plotting
    hist_plot = pd.concat(
        [
            df[[date_col, target_col]].rename(
                columns={date_col: "date", target_col: "value"}
            ),
            future_df.rename(columns={"date": "date", "prediction": "value"}),
        ],
        ignore_index=True,
    )

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df[date_col], df[target_col], label="historical")
    ax2.plot(
        future_df["date"], future_df["prediction"], label="forecast", linestyle="--"
    )
    ax2.set_title("Historical + Forecast")
    ax2.set_xlabel("Date")
    ax2.legend()
    st.pyplot(fig2)

    # Download forecasts
    csv_buf = io.StringIO()
    future_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode()
    st.download_button(
        "Download forecast CSV",
        data=csv_bytes,
        file_name="forecast.csv",
        mime="text/csv",
    )

    st.success("Done — forecast ready.")

# Footer / tips
st.markdown("---")
st.markdown(
    """
**Tips & notes**
- Ensure your data has consistent frequency (no big gaps). If you have missing days/weeks, consider resampling and filling.
- RandomForest is a strong baseline. For more advanced results, try XGBoost or specialized time series models (Prophet, ARIMA, SARIMAX).
- This app uses iterative forecasting: predicted values become inputs for future lags — common for short horizons.
"""
)
