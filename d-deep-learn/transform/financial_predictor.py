import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformer_model import SimpleTransformer

class FinancialDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        k = np.random.randint(0, self.X.shape[1] - self.seq_len - 3)
        return self.X[idx, k:k+self.seq_len, :], self.y[idx, k+self.seq_len-1]  # Return only last label

class FinancialPredictor:
    def __init__(self, ticker="BAP", period="60d", interval="2m"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        df = yf.download(self.ticker, period=self.period, interval=self.interval, progress=False)
        df = df.between_time("09:30", "16:00")  # Regular trading hours
        df["label"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        # Calculate steps per day and number of days
        steps_per_day = int(len(df) / len(df.index.normalize().unique()))
        n_days = len(df) // steps_per_day

        # Features: OHLCV
        cols = ["Open", "High", "Low", "Close", "Volume"]
        X1 = df[cols].values[:n_days * steps_per_day].reshape(n_days, steps_per_day, len(cols))
        y = df["label"].values[:n_days * steps_per_day].reshape(n_days, steps_per_day)

        # Scale features
        X = self.scaler.fit_transform(X1.reshape(-1, len(cols))).reshape(X1.shape)
        
        return X, y, steps_per_day

    def train(self, epochs=10, seq_len=32, batch_size=4, learning_rate=1e-3):
        X, y, steps_per_day = self.load_data()
        n_days = len(X)
        
        # Split data into train and validation
        train_size = int(0.8 * n_days)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Create datasets
        train_dataset = FinancialDataset(X_train, y_train, seq_len)
        val_dataset = FinancialDataset(X_val, y_val, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model and training components
        self.model = SimpleTransformer(in_features=5, d_model=32, nhead=4).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = self.model(xb)
                # Print shapes for understanding
                print(f"\nInput shape: {xb.shape}")
                print(f"Output shape: {output.shape}")
                print(f"Target shape: {yb.shape}")
                loss = loss_fn(output, yb)
                print(f"Loss value: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                break  # Just show first batch
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self.validate(val_loader, loss_fn)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            print("-" * 50)

    def validate(self, val_loader, loss_fn):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = self.model(xb)
                val_loss += loss_fn(output, yb).item()
                
                pred = torch.sigmoid(output) >= 0.5
                correct += (pred == yb).sum().item()
                total += yb.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        return avg_val_loss, accuracy

    def predict(self, sequence_data):
        """
        Make predictions on new data
        sequence_data: numpy array of shape (1, sequence_length, 5) containing OHLCV data
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        self.model.eval()
        with torch.no_grad():
            # Scale the input data
            scaled_data = self.scaler.transform(sequence_data.reshape(-1, 5)).reshape(sequence_data.shape)
            x = torch.FloatTensor(scaled_data).to(self.device)
            output = self.model(x)
            probabilities = torch.sigmoid(output)
            predictions = (probabilities >= 0.5).float()
            return predictions, probabilities

if __name__ == "__main__":
    predictor = FinancialPredictor()
    # Test with a very small batch to understand processing
    predictor.train(epochs=1, batch_size=3, learning_rate=1e-4)  # Small batch for demonstration