from data import result_df
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X = result_df[['Date']].astype('datetime64').astype(int).astype(float)
y = result_df[['Close']]
lm = LinearRegression()
lm.fit(X, y)
X_predict = lm.predict(X)

plt.scatter(X, y)
