import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from titans_pytorch import MemoryAsContextTransformer
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConsumptionPredictor:
    def __init__(self, 
                 num_tokens=256,
                 dim=256,  # Increased from 128
                 depth=2,   # Increased from 1
                 segment_len=128,
                 num_persist_mem_tokens=8,  # Increased from 4
                 num_longterm_mem_tokens=32):  # Increased from 16
        
        logger.info("Initializing model with full parameters...")
        self.scaler = MinMaxScaler(feature_range=(0, num_tokens-1))
        self.num_tokens = num_tokens
        
        self.transformer = MemoryAsContextTransformer(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            segment_len=segment_len,
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens
        )
        
    def prepare_data(self, excel_path, max_rows=10000): 
        logger.info(f"Loading data (max {max_rows} rows)...")
        
        self.df = pd.read_excel(excel_path, nrows=max_rows)
        logger.info(f"Initial dataframe shape: {self.df.shape}")
     
        logger.info("Processing timestamps...")
        self.df['datetime'] = pd.to_datetime(self.df['Day']) + \
                            pd.to_timedelta(self.df['Hour'], unit='h') + \
                            pd.to_timedelta(self.df['Minute'], unit='m')
        
        self.df = self.df.sort_values('datetime')
        
        logger.info("Scaling consumption values...")
        consumption_values = self.df['Consum'].values.reshape(-1, 1)
        consumption_scaled = self.scaler.fit_transform(consumption_values)
        token_ids = torch.tensor(consumption_scaled.astype(int)).squeeze()
        
        logger.info(f"Final token_ids shape: {token_ids.shape}")
        return token_ids
    
    def create_sequences(self, token_ids, seq_length=256, stride=64): 
        logger.info(f"Creating sequences from token_ids shape: {token_ids.shape}")
        sequences = []
        
        for i in tqdm(range(0, len(token_ids) - seq_length - 1, stride),
                     desc="Creating sequences"):
            seq = token_ids[i:i + seq_length]
            target = token_ids[i + 1:i + seq_length + 1]
            sequences.append((seq, target))
        
        logger.info(f"Created {len(sequences)} sequences (length={seq_length}, stride={stride})")
        return sequences

    def calculate_metrics(self, actual, predicted):
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        errors = actual - predicted
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        mae = np.mean(abs_errors)
      
        non_zero_mask = actual != 0
        mape = np.mean(np.abs(errors[non_zero_mask] / actual[non_zero_mask])) * 100
        
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        n = len(actual)
        p = 1  # nr de predictors
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'adjustedR2':adjusted_r2
        }
        
        return metrics

    def train(self, sequences, num_epochs=10, batch_size=8): 
        logger.info(f"Starting training with {len(sequences)} sequences")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        self.transformer = self.transformer.to(device)
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)
        
        epoch_losses = []
        
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            batch_losses = []
            
            for i in tqdm(range(0, len(sequences), batch_size), 
                         desc=f"Epoch {epoch+1} batches",
                         leave=False):
                batch_sequences = sequences[i:i + batch_size]
                x = torch.stack([seq[0] for seq in batch_sequences]).to(device)
                y = torch.stack([seq[1] for seq in batch_sequences]).to(device)
                
                loss = self.transformer(x, return_loss=True)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                
            avg_loss = np.mean(batch_losses)
            epoch_losses.append(avg_loss)
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        return epoch_losses
    
    def predict(self, initial_sequence, prediction_length=96):
        logger.info("Generating predictions...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            
            initial_sequence = initial_sequence.to(device)
            if initial_sequence.dim() == 1:
                initial_sequence = initial_sequence.unsqueeze(0)
            logger.info(f"Initial sequence shape: {initial_sequence.shape}")
            
            try:
                self.transformer = self.transformer.to(device)
                
                current_sequence = initial_sequence
                predictions = []
                
                for _ in range(prediction_length):
                   
                    output = self.transformer(current_sequence, return_loss=False)
                    next_token = output[:, -1:].argmax(dim=-1) 
                    predictions.append(next_token.item())
                 
                    current_sequence = torch.cat([
                        current_sequence[:, 1:],
                        next_token
                    ], dim=1)
                
                logger.info(f"Generated {len(predictions)} predictions")
                
                predictions_np = np.array(predictions).reshape(-1, 1)
                logger.info(f"Predictions shape before scaling: {predictions_np.shape}")
                
                
                predictions_scaled = self.scaler.inverse_transform(predictions_np)
                logger.info(f"Scaled predictions shape: {predictions_scaled.shape}")
                
                return predictions_scaled.squeeze()
                
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")
                logger.error(f"Initial sequence shape: {initial_sequence.shape}")
                raise

def main():
    logger.info("Starting full-scale test run...")
    start_time = datetime.now()
    
    predictor = ConsumptionPredictor()
    
    token_ids = predictor.prepare_data('Consum 2023-2024 OMEPA.xlsx', max_rows=10000)
    
    sequences = predictor.create_sequences(token_ids)
    
    epoch_losses = predictor.train(sequences)
    
    initial_seq = token_ids[:256]
    logger.info(f"Input sequence shape before prediction: {initial_seq.shape}")
    if len(initial_seq) == 0:
        raise ValueError("Empty input sequence")
    predictions = predictor.predict(initial_seq)
    
    actual_values = predictor.scaler.inverse_transform(
        token_ids[256:256+96].numpy().reshape(-1, 1)
    ).squeeze()
    
    metrics = predictor.calculate_metrics(actual_values, predictions)
    
    print("\nModel Performance Metrics:")
    print("=========================")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    print(f"R-squared (R2): {metrics['R2']:.4f}")
    
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(actual_values, label='Actual', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
    plt.fill_between(range(len(predictions)),
                     predictions - metrics['RMSE'],
                     predictions + metrics['RMSE'],
                     color='red', alpha=0.1, label='RMSE Band')
    plt.title('Consumption Prediction with Error Bands')
    plt.xlabel('Time Steps (15-minute intervals)')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    errors = actual_values - predictions
    plt.hist(errors, bins=30, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'training_time': total_time,
        'predictions': predictions,
        'actual_values': actual_values
    }

if __name__ == "__main__":
    main()
