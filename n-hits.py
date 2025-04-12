import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
def load_consumption_data(file_path):
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['Day'], format='%d/%m/%Y') + \
               pd.to_timedelta(df['Hour']-1, unit='h')
    df = df.sort_values('ds')
    df = df.drop_duplicates('ds')
    df['y'] = df['MWh']
    df = df.set_index('ds')
    full_index = pd.date_range(start=df.index.min(), 
                             end=df.index.max(), 
                             freq='1H')
    df = df.reindex(full_index)
    df['y'] = df['y'].interpolate(method='linear', limit=4)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['lag_24h'] = df['y'].shift(24)
    df['lag_48h'] = df['y'].shift(48)
    df['lag_168h'] = df['y'].shift(168)  
    df['rolling_mean_24h'] = df['y'].rolling(window=24).mean()
    df['rolling_mean_week'] = df['y'].rolling(window=168).mean()
    df = df.reset_index()
    df = df.rename(columns={'index': 'ds'})
    df = df.dropna()
    return df
class NHITSBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, stack_id=0):
        super(NHITSBlock, self).__init__()
        self.stack_id = stack_id
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.backcast_layer = nn.Linear(hidden_size, input_size)
        self.forecast_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        features = self.fc_stack(x)
        backcast = self.backcast_layer(features)
        forecast = self.forecast_layer(features)
        return backcast, forecast
class NHITS(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_blocks=4, num_stacks=3):
        super(NHITS, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.blocks = nn.ModuleList([
            NHITSBlock(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                stack_id=i % num_stacks
            ) for i in range(num_blocks)
        ])
    def forward(self, x):
        batch_size = x.size(0)
        x_input = x[:, -self.input_size:, 0].view(batch_size, -1)
        forecast = torch.zeros(batch_size, self.output_size, device=x.device)
        residuals = x_input
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.total_len = len(data)
        self.valid_indices = self.total_len - seq_length - pred_length + 1
    def __len__(self):
        return max(0, self.valid_indices)
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, 0]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)
def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    model.load_state_dict(best_model)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('nhits_training_loss.png')
    plt.close()
    return model, train_losses, val_losses
def evaluate_model(model, test_loader, scaler_y, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            actuals.append(target.cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(actuals, axis=0)
    if scaler_y is not None:
        y_pred_2d = y_pred.reshape(-1, 1)
        y_true_2d = y_true.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_2d).flatten()
        y_true = scaler_y.inverse_transform(y_true_2d).flatten()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    range_y = np.max(y_true) - np.min(y_true)
    nrmse = rmse / range_y
    mbe = np.mean(y_pred - y_true)
    cv = (rmse / np.mean(y_true)) * 100
    smape = np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    metrics = {
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
    print("\nForecast Metrics:")
    for metric, value in metrics.items():
        print(f"**{metric}: {value:.4f}**")
    return y_pred, y_true, metrics
def plot_forecast(df, test_dates, y_true, y_pred, history_window=168):
    plt.figure(figsize=(15, 8))
    history_start = max(0, len(df) - len(test_dates) - history_window)
    plt.plot(df['ds'].iloc[history_start:-len(test_dates)], 
             df['y'].iloc[history_start:-len(test_dates)], 
             label='Historical Data', alpha=0.7)
    plt.plot(test_dates, y_true, 
             linestyle='--', color='green', label='Actual Values')
    plt.plot(test_dates, y_pred, 
             color='red', label='NHITS Forecast', linewidth=2)
    plt.title('National Electricity Consumption Forecast - NHITS')
    plt.xlabel('Date')
    plt.ylabel('Consumption (MWh)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('nhits_forecast.png')
    plt.close()
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading data...")
    df = load_consumption_data("Consum National 2022-2024.csv")
    print(f"Loaded {len(df)} data points")
    seq_length = 168  
    pred_length = 24  
    input_size = seq_length  
    output_size = pred_length  
    features = ['y', 'hour', 'day_of_week', 'month', 'is_weekend', 
               'lag_24h', 'lag_48h', 'lag_168h', 'rolling_mean_24h']
    data = df[features].values
    scaler_X = StandardScaler()
    scaled_data = scaler_X.fit_transform(data)
    scaler_y = StandardScaler()
    y_data = df[['y']].values
    scaler_y.fit(y_data)
    train_size = int(0.7 * len(scaled_data))
    val_size = int(0.15 * len(scaled_data))
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size+val_size]
    test_data = scaled_data[train_size+val_size:]
    train_dataset = TimeSeriesDataset(train_data, seq_length, pred_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length, pred_length)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = NHITS(
        input_size=input_size,
        output_size=output_size,
        hidden_size=256,
        num_blocks=4,
        num_stacks=3
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print("Starting training...")
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        lr=0.001
    )
    torch.save(model.state_dict(), 'nhits_national_consumption.pth')
    print("Model saved to nhits_national_consumption.pth")
    print("Evaluating on test set...")
    y_pred, y_true, metrics = evaluate_model(model, test_loader, scaler_y, device)
    test_dates = df['ds'].iloc[train_size+val_size:train_size+val_size+len(y_true)]
    plot_forecast(df, test_dates, y_true, y_pred)
    print("\nFinal NHITS Forecast Metrics:")
    for key, value in metrics.items():
        print(f"**{key}: {value:.4f}**")
    return metrics
if __name__ == "__main__":
    main()
