import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping


np.random.seed(42)
tf.random.set_seed(42)

def calculate_metrics(y_true, y_pred):
    """Calculate all requested metrics for forecast evaluation."""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    nonzero_idx = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range > 0 else np.nan
    mbe = np.mean(y_pred - y_true)
    cv = rmse / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    smape = np.nan_to_num(smape)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'NRMSE': nrmse,
        'MBE': mbe,
        'CV': cv,
        'sMAPE': smape,
        'R2': r2
    }

def create_features(df):
    """Create time-based features from datetime index."""
    df_features = df.copy()
    
    df_features['hour'] = df_features.index.hour
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['dow_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['dow_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    df_features['lag_24'] = df_features['MWh'].shift(24)  # Previous day
    df_features['lag_168'] = df_features['MWh'].shift(168)  # Previous week
    
    df_features = df_features.dropna()
    
    return df_features

def build_model(input_shape, output_dim):
    """Build a simple but effective LSTM model."""
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    print("Robust Time Series Forecasting with Comprehensive Metrics")
    print("-------------------------------------------------------")
    
    print("Loading data...")
    df = pd.read_csv("Consum National 2022-2024.csv")
    print(f"Dataset shape: {df.shape}")
    
    print("\nPreprocessing data...")
    df['Date'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    df['Timestamp'] = df.apply(lambda row: row['Date'] + timedelta(hours=row['Hour']-1), axis=1)
    df = df.set_index('Timestamp')
    
    df = df[['MWh']]
    
    df_features = create_features(df)
    print(f"Features created. Data shape: {df_features.shape}")
    print(f"Features: {df_features.columns.tolist()}")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)
    
    sequence_length = 24 * 3  # 3 days (shorter to reduce memory usage)
    forecast_horizon = 24  # 1 day ahead
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - forecast_horizon + 1):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length:i+sequence_length+forecast_horizon, 0])
    
    X, y = np.array(X), np.array(y)
    print(f"Created sequences: X shape: {X.shape}, y shape: {y.shape}")
    
    test_split = 0.2
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    print("\nBuilding and training model...")
    model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    try:
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()
        
        print("Training history saved as 'training_history.png'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Trying with smaller batch size...")
        
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,  # Smaller batch size
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
    
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    mwh_position = 0
    n_features = scaled_data.shape[1]
    
    y_pred_template = np.zeros((y_pred.shape[0], y_pred.shape[1], n_features))
    y_pred_template[:, :, mwh_position] = y_pred
    
    y_pred_flat = y_pred_template.reshape(-1, n_features)
    
    y_pred_inverse = scaler.inverse_transform(y_pred_flat)[:, mwh_position]
    y_pred_inverse = y_pred_inverse.reshape(-1, forecast_horizon)
    
    y_test_template = np.zeros((y_test.shape[0], y_test.shape[1], n_features))
    y_test_template[:, :, mwh_position] = y_test
    
    y_test_flat = y_test_template.reshape(-1, n_features)
    
    y_test_inverse = scaler.inverse_transform(y_test_flat)[:, mwh_position]
    y_test_inverse = y_test_inverse.reshape(-1, forecast_horizon)
    
    print("\nCalculating forecast metrics...")
    metrics = calculate_metrics(y_test_inverse.flatten(), y_pred_inverse.flatten())
    
    print("\nForecast Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nComparing with baseline models...")
    
    persistence_pred = []
    for i in range(len(X_test)):
        last_value = X_test[i, -1, mwh_position]
        persistence_pred.append(np.repeat(last_value, forecast_horizon))
    persistence_pred = np.array(persistence_pred)
    
    persistence_template = np.zeros((persistence_pred.shape[0], persistence_pred.shape[1], n_features))
    persistence_template[:, :, mwh_position] = persistence_pred
    persistence_flat = persistence_template.reshape(-1, n_features)
    persistence_inverse = scaler.inverse_transform(persistence_flat)[:, mwh_position]
    persistence_inverse = persistence_inverse.reshape(-1, forecast_horizon)
    
    baseline_metrics = calculate_metrics(y_test_inverse.flatten(), persistence_inverse.flatten())
    
    print("\nBaseline (Persistence) Metrics:")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    improvement = (baseline_metrics['RMSE'] - metrics['RMSE']) / baseline_metrics['RMSE'] * 100
    print(f"\nImprovement over baseline (RMSE): {improvement:.2f}%")
    
    print("\nPlotting forecast results...")
    for i in range(3):
        sample_idx = np.random.randint(0, len(y_test_inverse))
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(forecast_horizon), y_test_inverse[sample_idx], 'b-', label='Actual')
        plt.plot(range(forecast_horizon), y_pred_inverse[sample_idx], 'r-', label='Forecast')
        plt.title(f'24-Hour Electricity Consumption Forecast (Sample {sample_idx})')
        plt.xlabel('Hours Ahead')
        plt.ylabel('Electricity Consumption (MWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'forecast_sample_{i+1}.png')
        plt.close()
    
    sample_idx = np.random.randint(0, len(y_test_inverse))
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(forecast_horizon), y_test_inverse[sample_idx], 'k-', label='Actual')
    plt.plot(range(forecast_horizon), y_pred_inverse[sample_idx], 'b-', label='LSTM Model')
    plt.plot(range(forecast_horizon), persistence_inverse[sample_idx], 'g--', label='Baseline (Persistence)')
    plt.title('Model Comparison: 24-Hour Forecast')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Electricity Consumption (MWh)')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison.png')
    plt.close()
    
    model.save('lstm_forecast_model.h5')
    print("\nForecasting completed successfully!")
    print("Model saved as 'lstm_forecast_model.h5'")

    metrics_labels = list(metrics.keys())
    model_metrics = [metrics[m] for m in metrics_labels]
    baseline_metrics_values = [baseline_metrics[m] for m in metrics_labels]
    
    fig, ax = plt.figure(figsize=(12, 10)), plt.subplot(111)
    y_pos = np.arange(len(metrics_labels))
    width = 0.35
    
    ax.barh(y_pos - width/2, model_metrics, width, label='LSTM Model')
    ax.barh(y_pos + width/2, baseline_metrics_values, width, label='Baseline')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_labels)
    ax.invert_yaxis() 
    ax.set_xlabel('Value')
    ax.set_title('Metrics Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()
    
    print("Metrics comparison saved as 'metrics_comparison.png'")

if __name__ == "__main__":
    main()
