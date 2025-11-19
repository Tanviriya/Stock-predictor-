import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure plots work everywhere
matplotlib.use("TkAgg")

# Try importing yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class StockPredictor:
    def __init__(self, symbol="AAPL", period="2y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.predictions = None

    def get_data(self):
        """Fetch data from Yahoo Finance or create demo data."""
        if HAS_YFINANCE:
            try:
                stock = yf.Ticker(self.symbol)
                self.data = stock.history(period=self.period)
                if len(self.data) > 0:
                    print(f"✓ Downloaded data for {self.symbol}")
                    return
                else:
                    print("⚠ Data empty, using demo data")
            except:
                print("⚠ Error downloading data, using demo data")
        else:
            print("⚠ yfinance not installed, using demo data")

        self._make_demo_data()

    def _make_demo_data(self):
        """Create demo stock data if yfinance is unavailable."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", "2024-07-01")

        price = 150
        prices = [price]
        for _ in range(len(dates) - 1):
            change = price * np.random.normal(0.001, 0.02)
            price = max(price + change, 1)
            prices.append(price)

        self.data = pd.DataFrame({
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
        }, index=dates)

        print("✓ Demo data created")

    def add_features(self):
        """Add technical indicators to the dataset."""
        df = self.data.copy()

        # Moving averages
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()

        # RSI calculation
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-6)                      # FIXED: Avoid division by zero
        df["RSI"] = 100 - (100 / (1 + rs))

        # Price changes
        df["Price_Change"] = df["Close"].pct_change()
        df["Volume_Change"] = df["Volume"].pct_change()

        # Lag features
        df["Yesterday"] = df["Close"].shift(1)
        df["Day_Before"] = df["Close"].shift(2)

        # Target variable (tomorrow's price)
        df["Target"] = df["Close"].shift(-1)

        # Remove NaN rows
        self.data = df.dropna()
        print(f"✓ Added features (Data Points: {len(self.data)})")

    def train(self):
        """Train the Random Forest model."""
        features = [
            "Close", "Volume", "High", "Low",
            "MA_5", "MA_20", "RSI",
            "Price_Change", "Volume_Change",
            "Yesterday", "Day_Before"
        ]

        X = self.data[features]
        y = self.data["Target"]

        # Time based split (do not shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Make predictions
        predictions = self.model.predict(X_test)

        # Store for plotting
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print("✓ Model Trained")
        print(f"  R² Score: {r2:.2%}")
        print(f"  RMSE Error: ${rmse:.2f}")

    def plot_results(self):
        """Plot actual vs predicted prices."""
        plt.figure(figsize=(12, 5))
        dates = self.data.index[-len(self.y_test):]

        plt.plot(dates, self.y_test.values, label="Actual Price", linewidth=2)
        plt.plot(dates, self.predictions, label="Predicted Price", linewidth=2, alpha=0.7)

        plt.title(f"{self.symbol} — Actual vs Predicted Prices")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show(block=True)           # FIX: Keeps window open

    def predict_tomorrow(self):
        """Predict tomorrow’s closing price."""
        features = [
            "Close", "Volume", "High", "Low",
            "MA_5", "MA_20", "RSI",
            "Price_Change", "Volume_Change",
            "Yesterday", "Day_Before"
        ]

        latest_data = self.data[features].iloc[-1:].values
        predicted_price = self.model.predict(latest_data)[0]
        current_price = self.data["Close"].iloc[-1]

        change = predicted_price - current_price
        percent = (change / current_price) * 100

        print("\n========================================")
        print(f"📈 Prediction for {self.symbol}")
        print("========================================")
        print(f"Current Price:     ${current_price:.2f}")
        print(f"Predicted Price:   ${predicted_price:.2f}")
        print(f"Change:            ${change:+.2f} ({percent:+.2f}%)")
        print("========================================\n")

        return predicted_price

    def run(self):
        """Run the entire prediction pipeline."""
        print(f"\n🚀 Running Stock Predictor for: {self.symbol}\n")

        self.get_data()
        self.add_features()
        self.train()
        self.plot_results()
        self.predict_tomorrow()


# Run the predictor
if __name__ == "__main__":
    predictor = StockPredictor("AAPL", period="2y")
    predictor.run()
