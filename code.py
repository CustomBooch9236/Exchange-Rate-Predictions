import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Set parameters from config file
symbol = config['YahooFinance']['symbol']
start_date = config['YahooFinance']['start_date']
end_date = config['YahooFinance']['end_date']
test_size = float(config['Model']['test_size'])
shuffle_data = bool(config['Model']['shuffle_data'])

# Download exchange rate data from Yahoo Finance
df = yf.download(symbol, start=start_date, end=end_date)

# Adding a new column with the lagged exchange rate
df.loc[:, 'Lagged_Rate'] = df['Exchange_Rate'].shift(1)

# Removing the first row since it doesn't have a lagged value
df = df.iloc[1:]

# Splitting the data into training and testing sets
X = df[['Lagged_Rate']]
y = df[['Exchange_Rate']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Creating a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the exchange rate
predicted_rate = lr.predict(X_test)

# Printing the predicted rate
print(predicted_rate[-1])
