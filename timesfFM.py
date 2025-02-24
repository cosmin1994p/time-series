import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_consumption_data(file_path):
    """
    Enhanced data loading with better preprocessing
    """
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Create proper datetime index
    df['ds'] = pd.to_datetime(df['Day']) + \
               pd.to_timedelta(df['Hour'], unit='h') + \
               pd.to_timedelta(df['Minute'], unit='m')
    
    # Sort and check for duplicates
    df = df.sort_values('ds')
    df = df.drop_duplicates('ds')
    
    # Handle missing values
    df['y'] = df['Consum']
    
    # Add unique_id
    df['unique_id'] = 'consumption_data'
    
    # Ensure continuous time index with 15-min frequency
    df = df.set_index('ds')
    full_index = pd.date_range(start=df.index.min(), 
                             end=df.index.max(), 
                             freq='15min')
    
    # Reindex and interpolate
    df = df.reindex(full_index)
    df['unique_id'] = df['unique_id'].fillna('consumption_data')
    
    # Handle missing values with linear interpolation
    df['y'] = df['y'].interpolate(method='linear', limit=4)
    
    # Reset index to get ds as column
    df = df.reset_index()
    df = df.rename(columns={'index': 'ds'})
    
    return df[['unique_id', 'ds', 'y']]

def initialize_model(model_version="2.0", backend="cpu"):
    """Enhanced model initialization with better parameters"""
    if model_version == "2.0":
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=96,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )
    else:
        raise ValueError("Currently only supporting model version 2.0")
    
    return tfm

def calculate_metrics(actual_values, forecast_values):
    """Calculate comprehensive forecast metrics"""
    # Basic error metrics
    mse = np.mean((actual_values - forecast_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - forecast_values))
    mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
    
    # Additional metrics
    nrmse = rmse / (actual_values.max() - actual_values.min())
    mbe = np.mean(forecast_values - actual_values)
    cv = (rmse / np.mean(actual_values)) * 100
    smape = np.mean(200 * np.abs(forecast_values - actual_values) / (np.abs(actual_values) + np.abs(forecast_values)))
    
    # R-squared
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    ss_res = np.sum((actual_values - forecast_values) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
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

def run_consumption_forecast(tfm, input_df, forecast_horizon=96):
    """
    Run forecast on consumption data with enhanced validation
    """
    print("\nPreparing data for forecasting...")
    print(f"Data range: {input_df['ds'].min()} to {input_df['ds'].max()}")
    print(f"Total points: {len(input_df)}")
    
    # Use all but forecast_horizon points for training
    context_length = len(input_df) - forecast_horizon
    training_df = input_df.iloc[:context_length].copy()
    
    print("\nRunning forecast...")
    forecast_df = tfm.forecast_on_df(
        inputs=training_df,
        freq="15min",
        value_name="y",
        num_jobs=1  # Avoid multiprocessing issues
    )
    
    # Get forecast column name
    forecast_col = [col for col in forecast_df.columns 
                   if col not in ['unique_id', 'ds', 'y']][0]
    
    print("\nCalculating metrics...")
    
    # Get the forecast values
    forecast_values = forecast_df[forecast_col].values
    
    # Get the corresponding actual values
    actual_values = input_df['y'].iloc[-len(forecast_values):].values
    
    # Remove any NaN values
    mask = ~np.isnan(actual_values) & ~np.isnan(forecast_values)
    actual_values = actual_values[mask]
    forecast_values = forecast_values[mask]
    
    # Calculate metrics if we have valid data
    if len(actual_values) > 0:
        metrics = calculate_metrics(actual_values, forecast_values)
        print("\nForecast Metrics:")
        for metric, value in metrics.items():
            print(f"**{metric}: {value:.3f}**")
    else:
        print("Warning: No valid data points for comparison")
        metrics = None
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(input_df['ds'][:context_length], 
             input_df['y'][:context_length],
             label='Historical Data', alpha=0.7)
    
    # Plot actual future data
    plt.plot(input_df['ds'][context_length:],
             input_df['y'][context_length:],
             linestyle='--', color='green',
             label='Actual Values', alpha=0.7)
    
    # Plot forecast
    plt.plot(forecast_df['ds'],
             forecast_df[forecast_col],
             color='red', label='TimesFM Forecast',
             linewidth=2)
    
    plt.title('Electricity Consumption Forecast')
    plt.xlabel('Time')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('consumption_forecast_enhanced.png')
    plt.close()
    
    return forecast_df, metrics

if __name__ == "__main__":
    print("Loading consumption data...")
    df = load_consumption_data("Consum 20232024 OMEPA.xlsx")
    print(f"Loaded {len(df)} data points")
    
    print("\nInitializing TimesFM model...")
    import torch
    backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using {backend} backend")
    
    # Initialize model
    tfm = initialize_model(model_version="2.0", backend=backend)
    
    # Run forecast
    forecast_df, metrics = run_consumption_forecast(tfm, df)
    print("\nForecast completed!")
