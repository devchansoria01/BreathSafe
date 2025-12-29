import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. AQI Calculation Logic (CPCB Standards) ---
AQI_BREAKPOINTS = {
    'pm2_5': [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 10000)],
    'pm10': [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 10000)],
    'no2': [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 10000)],
    'so2': [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 10000)],
    'co': [(0, 1.0), (1.1, 2.0), (2.1, 10.0), (10.1, 17.0), (17.1, 34.0), (34.1, 10000)],
    'o3': [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 10000)],
}
AQI_CATEGORIES = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]

def get_sub_index(value, pollutant):
    if pd.isna(value): return np.nan
    try:
        breakpoints = AQI_BREAKPOINTS[pollutant]
    except KeyError: return np.nan

    for i, (low, high) in enumerate(breakpoints):
        if low <= value <= high:
            aqi_low, aqi_high = AQI_CATEGORIES[i]
            sub_index = ((aqi_high - aqi_low) / (high - low)) * (value - low) + aqi_low
            return sub_index
    return 500 if value > breakpoints[-1][1] else 0

def calculate_aqi_from_pollutants(row):
    sub_indices = [
        get_sub_index(row.get('pm2_5_24h_avg'), 'pm2_5'),
        get_sub_index(row.get('pm10_24h_avg'), 'pm10'),
        get_sub_index(row.get('no2_24h_avg'), 'no2'),
        get_sub_index(row.get('so2_24h_avg'), 'so2'),
        get_sub_index(row.get('co_8h_avg_mg'), 'co'),
        get_sub_index(row.get('o3_8h_avg'), 'o3'),
    ]
    valid_indices = [idx for idx in sub_indices if pd.notna(idx)]
    return max(valid_indices) if valid_indices else np.nan

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def main():
    # --- 2. Data Loading & Preprocessing ---
    print("Step 1: Loading Delhi dataset...")
    try:
        df = pd.read_csv('delhi.csv', skiprows=3)
        df.columns = ['time', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co']
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').ffill().bfill()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- 3. Feature Engineering (Lags and Rolling Averages) ---
    print("Step 2: Calculating CPCB AQI and Lagged Features...")
    df['pm2_5_24h_avg'] = df['pm2_5'].rolling(24, min_periods=1).mean()
    df['pm10_24h_avg'] = df['pm10'].rolling(24, min_periods=1).mean()
    df['no2_24h_avg'] = df['no2'].rolling(24, min_periods=1).mean()
    df['so2_24h_avg'] = df['so2'].rolling(24, min_periods=1).mean()
    df['o3_8h_avg'] = df['o3'].rolling(8, min_periods=1).mean()
    df['co_8h_avg_mg'] = (df['co'].rolling(8, min_periods=1).mean()) / 1000.0
    df['AQI'] = df.apply(calculate_aqi_from_pollutants, axis=1)
    df = df.dropna()

    N_LAGS, N_FORECAST = 24, 72
    X_list = []
    for col in ['AQI', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co',]:
        for i in range(1, N_LAGS + 1):
            X_list.append(df[col].shift(i).rename(f'{col}_lag{i}'))
    
    X = pd.concat(X_list, axis=1)
    y = pd.concat([df['AQI'].shift(-i).rename(f'AQI_f{i}') for i in range(1, N_FORECAST + 1)], axis=1)
    
    full_df = pd.concat([X, y], axis=1).dropna()
    X, y = full_df[X.columns], full_df[y.columns]

    # --- 4. Training ---
    print("Step 3: Training Random Forest Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- 5. Accuracy & Metrics ---
    y_pred = model.predict(X_test)
    accuracies = []
    losses_rmse = []
    
    for i in range(N_FORECAST):
        mape = mean_absolute_percentage_error(y_test.iloc[:, i], y_pred[:, i])
        accuracies.append(max(0, (1 - mape) * 100))
        losses_rmse.append(np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])))

    print("\n" + "="*40)
    print(f"📊 MODEL PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Next-Hour Accuracy:      {accuracies[0]:.2f}%")
    print(f"72-Hour Accuracy:       {accuracies[-1]:.2f}%")
    print(f"Mean Validation Loss:    {np.mean(losses_rmse):.2f} (RMSE)")
    print("="*40)

    # --- 6. Future Prediction (Next 72 Hours) ---
    latest_input = X.iloc[[-1]]
    forecast_values = model.predict(latest_input)[0]
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=N_FORECAST, freq='H')

    forecast_df = pd.DataFrame({
        'Timestamp': forecast_dates,
        'Predicted_AQI': forecast_values.round(2)
    })
    forecast_df['Category'] = forecast_df['Predicted_AQI'].apply(get_aqi_category)

    print("\n📅 UPCOMING 72-HOUR PREDICTIONS (Sample):")
    print(forecast_df.head(10).to_string(index=False))

    # --- 7. Plotting Everything ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot A: Accuracy and Loss Curves
    ax1.plot(range(1, 73), accuracies, color='green', label='Accuracy (%)')
    ax1.set_ylabel('Accuracy %', color='green')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(1, 73), losses_rmse, color='red', label='Loss (RMSE)')
    ax1_twin.set_ylabel('Loss (RMSE)', color='red')
    ax1.set_title('Accuracy and Loss over Forecast Horizon')
    ax1.grid(True, alpha=0.3)

    # Plot B: Historical vs Future Forecast
    ax2.plot(df.index[-100:], df['AQI'][-100:], label='Historical AQI', color='blue')
    ax2.plot(forecast_df['Timestamp'], forecast_df['Predicted_AQI'], label='72h Forecast', color='orange', linestyle='--')
    ax2.axvline(x=last_date, color='black', linestyle=':', label='Current Time')
    ax2.set_title('Delhi AQI Trend: Past and Future')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
