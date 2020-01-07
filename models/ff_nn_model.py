from data import result_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import backend as K
from keras import optimizers
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate

# TODO: Scikit Learn for feature selection, for checking importance of polarity and sentiment in relation to predictions
# Keras wrapper that makes it compatible with scikit learn's need for an estimator object
# KerasClassifier or KerasRegressor is a wrapper for neural network that adds in estimator
# from keras.wrappers.scikit_learn import KerasRegressor
# Use features that remain after the RFE to train the model 

# Removing Average Mean, Differential to become a little more efficient
X = result_df[['Open', 'High', 'Low', 'Average Polarity', 'Polarity', 'Sentiment']]
y = result_df[['Close']]

# Time series specific train-test-split
tscv = TimeSeriesSplit(n_splits=5)

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

# Function for calculating r-squared as a metric for Keras
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def model_creation():
    # Begin NN
    model = Sequential()
    model.add(Dense(units = 20, activation = 'relu'))
    # TODO: Experiment with dropout
    model.add(Dense(units = 1))
    # TODO: Scheduled learning rate
    adam = optimizers.Adam(learning_rate = .003)
    model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics=[r_squared])
    return model

model = model_creation()
history = model.fit(X_train, y_train, epochs = 200, validation_data = (X_test, y_test))
score = model.evaluate(X_test, y_test, verbose = 0)

# Mean-absolute-error: 0.00567343344951354, R-squared: 0.953440248966217, after 200 epochs, lr = .003, n_splits=5
print("Mean-absolute-error: ", score[0])
print("R-squared: ", score[1])

# Plot of metrics and diagnostics - specifically of loss/val-loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Plot of metrics and diagnostics - specifically of r-squared/val-r-squared
plt.plot(history.history['r_squared'], label='train')
plt.plot(history.history['val_r_squared'], label='test')
plt.legend()
plt.show()
