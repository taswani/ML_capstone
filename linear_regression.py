from data import result_df
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#represent date in another way, dates are categorical (sklearn one-hot-encoding)
X = result_df[['Date']].astype('datetime64').astype(int).astype(float)
y = result_df[['Close']]
lm = LinearRegression()
#Train-test-split and crossvalidation - crossvalidate convenience function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
lm.fit(X_train, y_train)
#TODO: Look at different statistics over rolling windows, previous timestep's price, more of the info from past (basically add in more predictive features)
y_predict = lm.predict(X_test)
plt.scatter(X_train, y_train)
plt.plot(X_test, y_predict, color='red')
plt.show()
