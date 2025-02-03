import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def create_features(df):
    df = df.copy()
    
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # encoding ciclic (features reduse)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # lag features 
    df['load_lag24'] = df['RO Load'].shift(24)  # ziua anterioara
    df['load_lag168'] = df['RO Load'].shift(168)  # saptamana anterioara
    
    return df

class EfficientInformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super(EfficientInformer, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # mecanism de atentie simplificat 
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_embedding(x)
        
        # procesare in layers de atentie 
        for attention, norm in zip(self.attention_layers, self.layer_norms):
            # self-attention
            attn_out, _ = attention(x, x, x)
            x = self.dropout(attn_out)
            x = norm(x)
        
        # output
        return self.output_layer(x)

def train_efficient_model(df, batch_size=32, seq_length=168, pred_length=24, epochs=50):
    print("Preparing data...")

    df = create_features(df)
    
    feature_columns = [
        'RO Load', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'load_lag24', 'load_lag168'
    ]
    
    df = df.dropna()
    
    # scalare features
    scaler_dict = {}
    scaled_data = pd.DataFrame()
    
    for col in feature_columns:
        scaler = StandardScaler()
        scaled_data[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
        scaler_dict[col] = scaler
    
    # pregatire date 
    X = []
    y = []
    
    for i in range(len(scaled_data) - seq_length - pred_length + 1):
        X.append(scaled_data[feature_columns].iloc[i:i+seq_length].values)
        y.append(scaled_data['RO Load'].iloc[i+seq_length:i+seq_length+pred_length].values)
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y))
    
    # impartire date set de training set de test 
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # creare data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print("Initializing model...")
    model = EfficientInformer(
        input_dim=len(feature_columns),
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.1
    )
    
    criterion = nn.MSELoss()  # cat de departe sunt predictiile de valorile reale (MSE -mean squared error) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # tip de optimizator
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)# urmareste progresul modelului 
    
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs[:, -pred_length:, :], batch_y.unsqueeze(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # validare
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                val_outputs = model(batch_x)
                val_loss = criterion(val_outputs[:, -pred_length:, :], batch_y.unsqueeze(-1))
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # verificare de oprire 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'scalers': scaler_dict,
                'feature_columns': feature_columns
            }, 'efficient_informer_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    return model, scaler_dict, feature_columns

if __name__ == "__main__":
    # incarcare date
    print("Loading data...")
    df = pd.read_csv('Consum 2023-2024 National.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # training model 
    model, scalers, features = train_efficient_model(df)
    print("Training completed! Model saved as 'efficient_informer_model.pth'")
