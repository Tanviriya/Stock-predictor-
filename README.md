# Stock-predictor
# Stock Price Predictor 📈

A machine learning project that predicts stock prices using historical data and technical indicators.

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, SVM
- **Technical Indicators**: Moving Averages, MACD, RSI, Bollinger Bands
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Comprehensive Analysis**: Model comparison, feature importance, next-day predictions
- **Visualizations**: Price charts, prediction accuracy plots

## Quick Start


# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn yfinance

# Run the predictor
python stock_predictor.py


## Usage


from stock_predictor import StockPricePredictor

# Initialize predictor
predictor = StockPricePredictor(symbol='AAPL', period='2y')

# Run complete analysis
results, feature_importance, prediction = predictor.run_complete_analysis()


## Sample Output

- **Model Performance**: RMSE, R², MAE metrics
- **Next Day Prediction**: Predicted price and percentage change
- **Feature Importance**: Top indicators affecting price movements
- **Visualizations**: Actual vs predicted price charts

## Technologies Used

- **Python**: Core programming language
- **Scikit-learn**: Machine learning models
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Yahoo Finance API**: Real-time stock data

## Project Structure


stock-price-predictor/
├── stock_predictor.py    # Main implementation
├── README.md            # This file
└── requirements.txt     # Dependencies


## Results

The Random Forest model typically achieves the best performance with R² > 0.95 on most stocks, making it suitable for short-term price predictions.
