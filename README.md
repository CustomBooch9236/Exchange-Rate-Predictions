# Exchange-Rate-Predictions
This Python script downloads historical exchange rate data between USD and INR from Yahoo Finance, builds a linear regression model using scikit-learn, and uses the model to predict the exchange rate.

# Requirements
The following packages are required to run this script:

- pandas
- numpy
- scikit-learn
- yfinance

These can be installed using pip or another package manager:
``` pip install pandas numpy scikit-learn yfinance ```

# Usage

To use this script, create a config.ini file in the same directory as the script with the following contents:
```
[YahooFinance]
symbol = INR=X
start_date = 2015-01-01
end_date = 2023-04-28

[Model]
test_size = 0.2
shuffle_data = False
```
In this config file, you can set the symbol for the exchange rate you want to download from Yahoo Finance, as well as the start and end dates for the historical data.

You can also set parameters for the machine learning model, such as the test size and whether to shuffle the data before splitting it into training and testing sets.

To run the script, use the following command:
```
python exchange_rate_prediction.py
```

The script will read the parameters from the config.ini file, download historical exchange rate data from Yahoo Finance, build a linear regression model using scikit-learn, and predict the exchange rate.

The predicted exchange rate will be output to the console.

# License
This script is released under the MIT License. Feel free to use and modify it as you like.
