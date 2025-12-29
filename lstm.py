import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

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
    try: breakpoints = AQI_BREAKPOINTS[pollutant]
    except KeyError: return np.nan
    for i, (low, high) in enumerate(breakpoints):
        if low <= value <= high:
            aqi_low, aqi_high = AQI_CATEGORIES[i]
            return ((aqi_high - aqi_low) / (high - low)) * (value - low) + aqi_low
    if value > breakpoints[-1][1]: return 500
    return 0

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

def build_lstm_model(seq_len, num_features, forecast_horizon, lstm_units=64, dropout_rate=0.1):
    inputs = layers.Input(shape=(seq_len, num_features))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

def create_sequences(data_features, data_target, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data_features) - seq_length - forecast_horizon + 1):
        X.append(data_features[i:(i + seq_length)])
        y.append(data_target[(i + seq_length):(i + seq_length + forecast_horizon)])
    return np.array(X), np.array(y)

def main():
    print("Loading data...")
    df = pd.read_csv('delhi.csv', skiprows=3)
    # Correcting column mapping to actual CSV structure
    df.columns = ['time', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co']
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').ffill().bfill()
    
    print("Calculating historical AQI...")
    df['pm2_5_24h_avg'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    df['pm10_24h_avg'] = df['pm10'].rolling(window=24, min_periods=1).mean()
    df['no2_24h_avg'] = df['no2'].rolling(window=24, min_periods=1).mean()
    df['so2_24h_avg'] = df['so2'].rolling(window=24, min_periods=1).mean()
    df['o3_8h_avg'] = df['o3'].rolling(window=8, min_periods=1).mean()
    df['co_8h_avg_mg'] = (df['co'].rolling(window=8, min_periods=1).mean()) / 1000.0
    df['AQI'] = df.apply(calculate_aqi_from_pollutants, axis=1)
    df = df.dropna()
    
    print("Engineering features...")
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    all_features = ['AQI', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co', 'hour_of_day', 'day_of_week', 'month']
    target_col = 'AQI'
    n_features = len(all_features)
    N_LAGS, N_FORECAST = 24, 72
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    feature_scaler = StandardScaler()
    train_features_scaled = feature_scaler.fit_transform(train_df[all_features])
    test_features_scaled = feature_scaler.transform(test_df[all_features])
    
    target_scaler = StandardScaler()
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])
    
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled.ravel(), N_LAGS, N_FORECAST)
    X_test, y_test_scaled = create_sequences(test_features_scaled, test_target_scaled.ravel(), N_LAGS, N_FORECAST)
    
    print(f"Training on {len(X_train)} sequences...")
    model = build_lstm_model(seq_len=N_LAGS, num_features=n_features, forecast_horizon=N_FORECAST)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    print("\nEvaluating model...")
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test = target_scaler.inverse_transform(y_test_scaled)
    
    # Accuracy Calculation
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy_pct = (1 - mape) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Specific hours evaluation
    y_test_t1, y_pred_t1 = y_test[:, 0], y_pred[:, 0]
    y_test_t72, y_pred_t72 = y_test[:, -1], y_pred[:, -1]
    
    acc_t1 = (1 - mean_absolute_percentage_error(y_test_t1, y_pred_t1)) * 100
    acc_t72 = (1 - mean_absolute_percentage_error(y_test_t72, y_pred_t72)) * 100

    print("\n" + "="*40)
    print("📊 LSTM MODEL PERFORMANCE")
    print("="*40)
    print(f"Overall Accuracy:        {accuracy_pct:.2f}%")
    print(f"Overall RMSE:            {rmse:.2f} AQI")
    print(f"Forecast Accuracy (t+1):  {acc_t1:.2f}%")
    print(f"Forecast Accuracy (t+72): {acc_t72:.2f}%")
    print("="*40)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_t1, y_pred_t1, alpha=0.5, label='t+1 Forecast')
    min_val = min(y_test_t1.min(), y_pred_t1.min())
    max_val = max(y_test_t1.max(), y_pred_t1.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    plt.title('Data Fitting Map: Actual vs. Predicted AQI (t+1 hour)')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.legend()
    plt.grid(True)
    plt.savefig('aqi_lstm_actual_vs_predicted.png')
    
    # Forecast
    last_sequence = df[all_features].iloc[-N_LAGS:]
    last_sequence_scaled = feature_scaler.transform(last_sequence).reshape(1, N_LAGS, n_features)
    forecast_scaled = model.predict(last_sequence_scaled)
    forecast = target_scaler.inverse_transform(forecast_scaled).flatten()
    
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=N_FORECAST, freq='h')
    forecast_df = pd.DataFrame({'Timestamp': forecast_dates, 'Predicted_AQI': forecast.round(2)})
    
    print("\nAQI Forecast (Next 72 Hours - First 10 rows):")
    print(forecast_df.head(10).to_string(index=False))

main()
