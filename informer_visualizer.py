import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
from efficient_informer import EfficientInformer  

def evaluate_model():
    print("Loading model and data...")
    checkpoint = torch.load('/Users/cosmin/Desktop/analiza articole/electricitate 2024/consum/informer models/efficient_informer_model.pth')
    df = pd.read_csv('/Users/cosmin/Desktop/analiza articole/electricitate 2024/consum/Consum 2023-2024 National.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    feature_columns = checkpoint['feature_columns']
    scalers = checkpoint['scalers']
    
    model = EfficientInformer(
        input_dim=len(feature_columns),
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    def create_features(df):
        df = df.copy()
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        df['load_lag24'] = df['RO Load'].shift(24)
        df['load_lag168'] = df['RO Load'].shift(168)
        return df
    
    print("Processing data...")
    df = create_features(df)
    df = df.dropna()

    scaled_data = pd.DataFrame()
    for col in feature_columns:
        scaler = scalers[col]
        scaled_data[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
    
    seq_length = 168
    pred_length = 24
    test_size = 168  
    
    predictions = []
    actuals = []
    dates = []
    
    print("Making predictions...")

    for i in range(0, test_size, pred_length):
        if i + seq_length + pred_length > len(scaled_data):
            break
            
     
        input_seq = scaled_data[feature_columns].iloc[i:i+seq_length].values
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)
        
     
        actual = df['RO Load'].iloc[i+seq_length:i+seq_length+pred_length].values
        pred_dates = df['date'].iloc[i+seq_length:i+seq_length+pred_length]
        
     
        with torch.no_grad():
            output = model(input_tensor)
            pred = output[:, -pred_length:, :].numpy().squeeze()
            pred = pred.reshape(-1, 1)
            pred = scalers['RO Load'].inverse_transform(pred).flatten()
        
        predictions.extend(pred)
        actuals.extend(actual)
        dates.extend(pred_dates)
    
    print("Calculating metrics...")
   
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print("\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    print("\nCreating visualizations...")

    plt.figure(figsize=(15, 8))
    plt.plot(dates, actuals, label='Actual', color='blue')
    plt.plot(dates, predictions, label='Predicted', color='red', linestyle='--')
    plt.title('Energy Load Prediction vs Actual (Last 7 Days)')
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()
    
    errors = predictions - actuals
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (MW)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.close()
    
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': errors,
        'Absolute_Error': np.abs(errors),
        'Percentage_Error': np.abs(errors/actuals) * 100
    })
    results_df.to_csv('detailed_results.csv', index=False)
    
    
    print("\nError Analysis:")
    print(f"Mean Error: {np.mean(errors):.2f} MW")
    print(f"Error Std Dev: {np.std(errors):.2f} MW")
    print(f"Max Absolute Error: {np.max(np.abs(errors)):.2f} MW")
    print(f"95th Percentile Error: {np.percentile(np.abs(errors), 95):.2f} MW")
    
    return results_df

if __name__ == "__main__":
    print("Starting model evaluation...")
    results = evaluate_model()
    print("\nEvaluation completed!")
    print("Results saved to 'detailed_results.csv'")
    print("Visualizations saved as 'prediction_results.png' and 'error_distribution.png'")