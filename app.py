import streamlit as st
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
import xlsxwriter
import warnings
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

st.markdown(
    """
    <style>
    /* Hide Streamlit header */
    header {visibility: hidden;}

    /* Remove top padding */
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Load and clean data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.dropna(subset=['timestamp', 'volume'])
    df['month_num'] = range(len(df))
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['quarter'] = df['timestamp'].dt.quarter
    return df

# Generate future features dynamically
def generate_future_dataframe(df, n_future=24):
    last_month_num = df['month_num'].iloc[-1]
    future_months = range(last_month_num + 1, last_month_num + 1 + n_future)
    future_dates = pd.date_range(start=df['timestamp'].max() + pd.DateOffset(months=1), periods=n_future, freq='MS')
    future_df = pd.DataFrame({
        'month_num': future_months,
        'month': future_dates.month,
        'year': future_dates.year,
        'quarter': future_dates.quarter
    })
    return future_df


# Accuracy metrics
def get_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# Forecasting models
def forecast_arima(df):
    model = ARIMA(df['volume'], order=(1,1,1)).fit()
    forecast = model.forecast(steps=24)
    future_dates = pd.date_range(start=df['timestamp'].max() + pd.DateOffset(months=1), periods=24, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
    y_pred = model.predict(start=0, end=len(df)-1)
    return forecast_df, y_pred


def forecast_rf(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = RandomForestRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future[['month_num', 'month', 'year', 'quarter']])
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_xgb(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = xgb.XGBRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future[['month_num', 'month', 'year', 'quarter']])
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_lgb(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = lgb.LGBMRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future[['month_num', 'month', 'year', 'quarter']])
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_lstm(df):
    data = df['volume'].values.reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(3, len(data_scaled)):
        X.append(data_scaled[i-3:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    last_input = data_scaled[-3:].reshape(1,3,1)
    forecast = []
    for _ in range(24):
        pred = model.predict(last_input)[0][0]
        forecast.append(pred)
        last_input = np.append(last_input[:,1:,:], [[[pred]]], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = scaler.inverse_transform(model.predict(X)).flatten()
    return forecast_df, y_pred

def forecast_sarima(df):
    model = SARIMAX(df['volume'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
    forecast = model.forecast(steps=24)
    future_dates = pd.date_range(df['timestamp'].max() + pd.DateOffset(months=1), periods=24, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
    y_pred = model.predict(start=0, end=len(df)-1)
    return forecast_df, y_pred

def forecast_ets(df):
    model = ExponentialSmoothing(df['volume'], seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(24)
    future_dates = pd.date_range(df['timestamp'].max() + pd.DateOffset(months=1), periods=24, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
    y_pred = model.fittedvalues
    return forecast_df, y_pred

def forecast_var(df):
    df_var = df[['volume', 'month']].dropna()
    model = VAR(df_var)
    results = model.fit(2)
    forecast = results.forecast(df_var.values[-2:], steps=24)
    future_dates = pd.date_range(df['timestamp'].max() + pd.DateOffset(months=1), periods=24, freq='MS')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast[:,0]})
    y_pred = results.fittedvalues['volume']
    return forecast_df, y_pred

# def forecast_catboost(df):
#     X = df[['month_num', 'month', 'year', 'quarter']]
#     y = df['volume']
#     model = CatBoostRegressor(verbose=0).fit(X, y)
#     future = generate_future_dataframe(df)
#     forecast = model.predict(future)
#     forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
#     y_pred = model.predict(X)
#     return forecast_df, y_pred

def forecast_gb(df):
    X = df[['month_num', 'month', 'year', 'quarter']]
    y = df['volume']
    model = GradientBoostingRegressor().fit(X, y)
    future = generate_future_dataframe(df)
    forecast = model.predict(future)
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': forecast})
    y_pred = model.predict(X)
    return forecast_df, y_pred

def forecast_hybrid_arima_xgb(df):
    arima_model = ARIMA(df['volume'], order=(1,1,1)).fit()
    arima_pred = arima_model.predict(start=0, end=len(df)-1)
    residuals = df['volume'] - arima_pred
    X = df[['month_num', 'month', 'year', 'quarter']]
    xgb_model = xgb.XGBRegressor().fit(X, residuals)
    future = generate_future_dataframe(df)
    xgb_forecast = xgb_model.predict(future)
    arima_forecast = arima_model.forecast(steps=24)
    final_forecast = arima_forecast + xgb_forecast
    forecast_df = pd.DataFrame({'ds': pd.date_range(df['timestamp'].max()+pd.DateOffset(months=1), periods=24, freq='MS'), 'yhat': final_forecast})
    y_pred = arima_pred + xgb_model.predict(X)
    return forecast_df, y_pred

# Export to Excel
def export_to_excel(results, output_path):
    workbook = xlsxwriter.Workbook(output_path)
    summary = workbook.add_worksheet('Summary')
    summary.write_row(0, 0, ['Model', 'Grade', 'RMSE', 'Explanation'])

    sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['RMSE'])
    for rank, (name, data) in enumerate(sorted_models, start=1):
        grade = f"{rank}st" if rank == 1 else f"{rank}nd" if rank == 2 else f"{rank}rd" if rank == 3 else f"{rank}th"
        explanation = f"{name} ranked {grade} based on RMSE of {round(data['metrics']['RMSE'],2)}. It performs well for {data['category']}."
        summary.write_row(rank, 0, [name, grade, data['metrics']['RMSE'], explanation])

        sheet = workbook.add_worksheet(name[:31])
        sheet.write_row(0, 0, ['Date', 'Forecasted Volume'])
        for i, row in enumerate(data['forecast'].itertuples(), start=1):
            sheet.write(i, 0, str(row.ds))
            sheet.write(i, 1, row.yhat)

    workbook.close()

# Streamlit UI

clock_html = """
<style>
#clock-box {
  position: fixed;
  top: 50px;
  left: 10px;
  z-index: 9999;
  font-family: 'Segoe UI', sans-serif;
  font-size: 20px;
  color: black;
  text-align: left;
  background-color: rgba(255,255,255,0.8);
  padding: 10px 15px;
  border-radius: 12px;
  box-shadow: 0 0 10px rgba(0,0,0,0.3);
}

@keyframes colorShift {
  0%   { color: #39ff14; }
  25%  { color: #00ffff; }
  50%  { color: #1e90ff; }
  75%  { color: #ff00ff; }
  100% { color: #39ff14; }
}

.animated-number {
  animation: colorShift 10s infinite;
  font-weight: bold;
}
</style>

<div id="clock-box">
  <div id="clock"></div>
  <div style="font-size: 14px; margin-top: 5px;">
    Got any Question/Suggestion?<br>
    Feel free to contact me at <span style="color: #00008B; font-weight: bold;">gurpreetsinghwfm@gmail.com</span><br>
    or WhatsApp me at <span class="animated-number">+91-8377001181</span>
  </div>
</div>

<script>
function updateClock() {
  const now = new Date();
  const timeString = now.toLocaleTimeString();
  document.getElementById('clock').textContent = timeString;
}
setInterval(updateClock, 1000);
updateClock();
</script>
"""
st.markdown(clock_html, unsafe_allow_html=True)




st.title("📊 AI - Forecasting Tool By Data Quest")

st.markdown(
    "<p style='color:#1f77b4; font-size:14px; margin-top:-10px;'>"
    "A tool developed by Gurpreet Singh"
    "</p>",
    unsafe_allow_html=True
)

# Code to download the sample CSV file 

sample_df = pd.DataFrame({
    "timestamp": [
        "01-01-2020","01-02-2020","01-03-2020","01-04-2020","01-05-2020","01-06-2020",
        "01-07-2020","01-08-2020","01-09-2020","01-10-2020","01-11-2020","01-12-2020",
        "01-01-2021","01-02-2021","01-03-2021","01-04-2021","01-05-2021","01-06-2021",
        "01-07-2021","01-08-2021","01-09-2021","01-10-2021","01-11-2021","01-12-2021",
        "01-01-2022","01-02-2022","01-03-2022","01-04-2022","01-05-2022","01-06-2022",
        "01-07-2022","01-08-2022","01-09-2022","01-10-2022","01-11-2022","01-12-2022",
        "01-01-2023","01-02-2023","01-03-2023","01-04-2023","01-05-2023","01-06-2023",
        "01-07-2023","01-08-2023","01-09-2023","01-10-2023","01-11-2023","01-12-2023",
        "01-01-2024","01-02-2024","01-03-2024","01-04-2024","01-05-2024","01-06-2024",
        "01-07-2024","01-08-2024","01-09-2024","01-10-2024","01-11-2024","01-12-2024"
    ],
    "volume": [
        14500000,15000000,14000000,13500000,13800000,14200000,
        14800000,15000000,16000000,16500000,17000000,17500000,
        15000000,15500000,16000000,15800000,16200000,16500000,
        17000000,17200000,18500000,19000000,19500000,20000000,
        16000000,16500000,17000000,16800000,17200000,17500000,
        18000000,18200000,19500000,20000000,20500000,21000000,
        18500000,19000000,19500000,18800000,19200000,19600000,
        20000000,20500000,21000000,21500000,22500000,23500000,
        19000000,19500000,20000000,19300000,19700000,20200000,
        20700000,21200000,21800000,22500000,23500000,24500000
    ]
})

sample_csv = sample_df.to_csv(index=False)

st.download_button(
    label="📥 Download Sample CSV",
    data=sample_csv,
    file_name="sample_forecast_template.csv",
    mime="text/csv"
)

st.info("Please download the sample CSV, update your historical data, and then upload it below.")

#Upload sample CSV file 

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    models = {
        'ARIMA': (forecast_arima, 'short-term, interpretable'),
        'SARIMA': (forecast_sarima, 'seasonal, interpretable'),
        'Exponential Smoothing': (forecast_ets, 'seasonal smoothing'),
        'VAR': (forecast_var, 'multivariate time series'),
        'Random Forest': (forecast_rf, 'short-term, interpretable'),
        'XGBoost': (forecast_xgb, 'high accuracy, nonlinear'),
        'LightGBM': (forecast_lgb, 'high accuracy, nonlinear'),
        # 'CatBoost': (forecast_catboost, 'nonlinear, categorical support'),
        'Gradient Boosting': (forecast_gb, 'nonlinear, interpretable'),
        'Hybrid ARIMA + XGBoost': (forecast_hybrid_arima_xgb, 'hybrid linear + nonlinear')
    }



    results = {}
    for name, (func, category) in models.items():
        forecast_df, y_pred = func(df)
        metrics = get_metrics(df['volume'], y_pred)
        results[name] = {
            'forecast': forecast_df,
            'y_pred': y_pred,
            'metrics': metrics,
            'category': category
        }

    # Show actual vs forecast for each model
        # Show actual vs forecast for each model


    st.subheader("📉 Actual vs Forecast")
    for name in models.keys():
        st.write(f"**{name}**")
        y_pred = results[name]['y_pred']
        forecast_df = results[name]['forecast']

        # Actuals
        actual_df = df[['timestamp', 'volume']].copy()
        actual_df.columns = ['ds', 'y']
        actual_df['source'] = 'Actual'

        # Forecasts
        forecast_df = forecast_df.copy()
        forecast_df['source'] = 'Forecast'
        forecast_df['y'] = forecast_df['yhat']

        # Combine actuals and forecast
        combined_df = pd.concat([actual_df, forecast_df], ignore_index=True)

        # Plot
        plt.figure(figsize=(12, 4))
        for src, group in combined_df.groupby('source'):
            plt.plot(group['ds'], group['y'],
                     label=src,
                     linestyle='--' if src == 'Forecast' else '-',
                     marker='o' if src == 'Actual' else None)

        plt.title(f"{name} - Actual vs Forecast")
        plt.xlabel("Month-Year")
        plt.ylabel("Volume")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Export to Excel
    export_to_excel(results, "forecast_summary.xlsx")
    with open("forecast_summary.xlsx", "rb") as f:
        st.download_button("📥 Download Forecast Excel", f, "forecast_summary.xlsx")
    
    st.success("✅ Forecast complete. Models ranked by RMSE in the summary sheet.")


