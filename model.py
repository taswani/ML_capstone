from data import result_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate

X = result_df[['Open', 'High', 'Low', 'Average Mean', 'Polarity', 'Sentiment', 'Differential']]
y = result_df[['Close']]

reg = xgb.XGBRegressor(n_estimators=50)
#Train-test-split and crossvalidation - crossvalidate convenience function (gives the type of error you want)
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train

predict = reg.predict(X_test)
print(reg.score(X_test, y_test))
# Best R-squared value I have so far is: -5.070259152080655e+24

# xgb.plot_importance(reg)
# plt.show()
