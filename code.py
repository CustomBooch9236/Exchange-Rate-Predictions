import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Yahoo Finance API parameters
symbol = config['YahooFinance']['symbol']
start_date = config['YahooFinance']['start_date']
end_date = config['YahooFinance']['end_date']

# Machine learning model parameters
test_size = float(config['Model']['test_size'])
shuffle_data = bool(config['Model']['shuffle_data'])

# Download exchange rate data from Yahoo Finance
df = yf.download(symbol, start=start_date, end=end_date)

# Create a new DataFrame with only the exchange rate column and date index
df = df.loc[:, ['Close']]
df.index.names = ['Date']
df.columns = ['Exchange_Rate']

# Add a new column with the lagged exchange rate
df['Lagged_Rate'] = df['Exchange_Rate'].shift(1)

# Drop rows with missing values
df.dropna(inplace=True)

# Split the data into training and testing sets
X = df.loc[:, ['Lagged_Rate']]
y = df.loc[:, 'Exchange_Rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle_data)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the exchange rate
last_exchange_rate = df.iloc[-1]['Exchange_Rate']
predicted_exchange_rate = model.predict([[last_exchange_rate]])
print(f'Predicted exchange rate: {predicted_exchange_rate[0]:.2f}')
