from data import result_df
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#represent date in another way, dates are categorical (sklearn one-hot-encoding)
X = result_df[['Date']].astype('datetime64').astype(int).astype(float)
y = result_df[['Close']]
lm = LinearRegression()
#Train-test-split and crossvalidation - crossvalidate convenience function
lm.fit(X, y)

plt.scatter(X, y)
plt.show()
