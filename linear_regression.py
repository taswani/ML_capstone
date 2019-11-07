from data import result_df
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

X = result_df[['Sentiment', 'Average Mean']]
y = result_df[['Close']]

lm = LinearRegression()
#Train-test-split and crossvalidation - crossvalidate convenience function (gives the type of error you want)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
lm.fit(X_train, y_train)

y_predict = lm.predict(X_test)

print(lm.score(X_test, y_test))

# plt.scatter(X_train, y_train)
# plt.plot(X_test, y_predict, color='red')
# plt.show()

# TODO: Measure performance via cross validation (K-Fold: 3 or 5 Folds) --> Time series split for cross-validation folds
# After setting up folds look to try to Random forests (XGBoosts) or other classification models

# Time series might cause data leakage in cross validation
# Brush up on R^2, mean-squared-error, mean-absolute-error
