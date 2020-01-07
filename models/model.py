from data import result_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate

X = result_df[['Open', 'High', 'Low', 'Average Polarity', 'Polarity', 'Sentiment']]
y = result_df[['Close']]

reg = xgb.XGBRegressor(n_estimators=50)
#Train-test-split and crossvalidation - crossvalidate convenience function (gives the type of error you want)
tscv = TimeSeriesSplit(n_splits=4)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Scaling all values for a normalized input and output
min_max_scaler = MinMaxScaler()

# Changed fit_transform to transform for test fold to avoid data leakage from future test set
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
y_train = min_max_scaler.fit_transform(y_train)
y_test = min_max_scaler.transform(y_test)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train

predict = reg.predict(X_test)
print("R-squared value: ", reg.score(X_test, y_test))
# Best R-squared value I have so far is: 0.9866523109744227

xgb.plot_importance(reg)
plt.show()

# TODO: RFE on classical model, and then use features selected to train neural network
# TODO: Yellow brick package for visualizing the feature elimination from RFE

lr = LinearRegression()

stregr = StackingRegressor(regressors=[lr, reg],
                           meta_regressor=reg)
stregr = stregr.fit(X_train, y_train)
print('Variance Score: %.4f' % stregr.score(X_train, y_train))