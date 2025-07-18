import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# For advanced models
try:
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    print("Some advanced models may not be available")

# For fetching stock data (you'll need to install: pip install yfinance)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available. Using synthetic data for demonstration.")

class StockPricePredictor:
    def __init__(self, symbol='AAPL', period='2y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance or create synthetic data"""
        if YFINANCE_AVAILABLE:
            try:
                stock = yf.Ticker(self.symbol)
                self.data = stock.history(period=self.period)
                print(f"Successfully fetched data for {self.symbol}")
                return True
            except Exception as e:
                print(f"Error fetching data: {e}")
                return False
        else:
            # Create synthetic stock data for demonstration
            print("Creating synthetic stock data for demonstration...")
            self._create_synthetic_data()
            return True
    
    def _create_synthetic_data(self):
        """Create synthetic stock data for demonstration purposes"""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2024-07-01', freq='D')
        
        # Create realistic stock price movement
        initial_price = 150
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            price_change = prices[-1] * returns[i]
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 1))  # Ensure price stays positive
        
        # Create OHLC data
        highs = [p * np.random.uniform(1.0, 1.05) for p in prices]
        lows = [p * np.random.uniform(0.95, 1.0) for p in prices]
        opens = [p * np.random.uniform(0.98, 1.02) for p in prices]
        volumes = np.random.randint(1000000, 10000000, len(dates))
        
        self.data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        print(f"Created synthetic data with {len(self.data)} records")
    
    def create_technical_indicators(self):
        """Create technical indicators as features"""
        df = self.data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        self.data = df
        print("Technical indicators created successfully")
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        # Select feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Width', 'BB_Position',
            'Price_Change', 'High_Low_Ratio', 'Volume_Ratio',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_5'
        ]
        
        # Remove rows with NaN values
        df_clean = self.data.dropna()
        
        self.features = df_clean[feature_columns]
        self.target = df_clean['Target']
        self.feature_names = feature_columns
        
        print(f"Features prepared: {self.features.shape[0]} samples, {self.features.shape[1]} features")
    
    def train_models(self):
        """Train multiple machine learning models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        
        # Add SVM if available
        try:
            models['SVM'] = SVR(kernel='rbf', C=100, gamma=0.1)
        except:
            pass
        
        # Train models
        print("Training models...")
        for name, model in models.items():
            print(f"Training {name}...")
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled if name != 'Random Forest' else X_train, y_train)
            
            # Make predictions
            if name == 'Random Forest':
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results = []
        for name, model_info in self.models.items():
            results.append({
                'Model': name,
                'RMSE': model_info['rmse'],
                'MAE': model_info['mae'],
                'R²': model_info['r2']
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE')
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_predictions(self):
        """Plot actual vs predicted prices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (name, model_info) in enumerate(self.models.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Plot actual vs predicted
                ax.scatter(self.y_test, model_info['predictions'], alpha=0.6)
                ax.plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Price')
                ax.set_ylabel('Predicted Price')
                ax.set_title(f'{name}\nR² = {model_info["r2"]:.4f}')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(self.models), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_predictions(self):
        """Plot time series predictions"""
        # Get the best model (lowest RMSE)
        best_model_name = min(self.models.keys(), 
                            key=lambda x: self.models[x]['rmse'])
        
        best_model_info = self.models[best_model_name]
        
        # Create time series plot
        plt.figure(figsize=(15, 8))
        
        # Get the last portion of data for visualization
        test_dates = self.data.index[-len(self.y_test):]
        
        plt.plot(test_dates, self.y_test.values, label='Actual Price', linewidth=2)
        plt.plot(test_dates, best_model_info['predictions'], 
                label=f'Predicted Price ({best_model_name})', linewidth=2)
        
        plt.title(f'{self.symbol} Stock Price Prediction - {best_model_name}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """Show feature importance for tree-based models"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            importance = rf_model.feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Plot top 15 features
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        else:
            print("Random Forest model not available for feature importance analysis")
            return None
    
    def predict_next_day(self):
        """Predict next day's closing price"""
        # Get the best model
        best_model_name = min(self.models.keys(), 
                            key=lambda x: self.models[x]['rmse'])
        
        best_model = self.models[best_model_name]['model']
        
        # Get the latest data point
        latest_features = self.features.iloc[-1:].values
        
        # Scale if needed
        if best_model_name != 'Random Forest':
            latest_features = self.scaler.transform(latest_features)
        
        # Make prediction
        prediction = best_model.predict(latest_features)[0]
        current_price = self.data['Close'].iloc[-1]
        
        print(f"\n{self.symbol} Stock Price Prediction")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Next Day Price: ${prediction:.2f}")
        print(f"Predicted Change: ${prediction - current_price:.2f} ({((prediction - current_price) / current_price) * 100:.2f}%)")
        print(f"Best Model Used: {best_model_name}")
        
        return prediction
    
    def run_complete_analysis(self):
        """Run the complete stock prediction analysis"""
        print(f"Starting Stock Price Prediction Analysis for {self.symbol}")
        print("="*60)
        
        # Step 1: Fetch data
        if not self.fetch_data():
            print("Failed to fetch data. Exiting.")
            return
        
        # Step 2: Create technical indicators
        self.create_technical_indicators()
        
        # Step 3: Prepare features
        self.prepare_features()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        results = self.evaluate_models()
        
        # Step 6: Visualize results
        self.plot_predictions()
        self.plot_time_series_predictions()
        
        # Step 7: Feature importance
        feature_importance = self.feature_importance()
        
        # Step 8: Next day prediction
        next_day_prediction = self.predict_next_day()
        
        return results, feature_importance, next_day_prediction

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = StockPricePredictor(symbol='AAPL', period='2y')
    
    # Run complete analysis
    results, feature_importance, prediction = predictor.run_complete_analysis()
    
    # You can also run individual components:
    # predictor.fetch_data()
    # predictor.create_technical_indicators()
    # predictor.prepare_features()
    # predictor.train_models()
    # predictor.evaluate_models()
    # predictor.plot_predictions()
    # predictor.predict_next_day()