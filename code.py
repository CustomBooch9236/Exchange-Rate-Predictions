import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf

# Downloading historical data from yahoo finance
df = yf.download('INR=X', start='2015-01-01', end='2023-04-28')

# Creating a new column with the exchange rate between USD and INR
df['Exchange_Rate'] = 1/df['Close']

# Creating a new DataFrame with only the exchange rate column and date index
df = df.loc[:, ['Exchange_Rate']]
df.index.names = ['Date']

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
