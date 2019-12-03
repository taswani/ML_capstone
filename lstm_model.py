from data import result_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import backend as K
from keras import optimizers
from keras.callbacks.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate

# Removing Average Mean, Differential to become a little more efficient
X = result_df[['Open', 'High', 'Low', 'Polarity', 'Sentiment']]
y = result_df[['Close']]

# Time series specific train-test-split
tscv = TimeSeriesSplit(n_splits=4)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# print("X_Train: ", X_train, "X_Test: ", X_test, "y_train: ", y_train, "y_test: ", y_test)

# Scaling all values for a normalized input and output
min_max_scaler = MinMaxScaler()

# Changed fit_transform to transform for test fold to avoid data leakage from future test set
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
y_train = min_max_scaler.fit_transform(y_train)
y_test = min_max_scaler.transform(y_test)

# Using keras' timeseriesgenerator in order to divide the data into batches
# Putting data into 3D for input to the LSTM
data_gen_train = TimeseriesGenerator(X_train, y_train,
                               length=14, sampling_rate=1,
                               batch_size=160)

data_gen_test = TimeseriesGenerator(X_test, y_test,
                               length=14, sampling_rate=1,
                               batch_size=160)

# Done for Input shape of LSTM
train_X, train_y = data_gen_train[0]
test_X, test_y = data_gen_test[0]

# (32, 7, 5)
# print(train_X.shape)
# (32, 1)
# print(train_y)

def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# TODO: feed forward neural network instead

# Begin LSTM

model = Sequential()
# Remove dropout initially to let overfitting happen.

model.add(LSTM(units = 40, return_sequences = False, input_shape = (train_X.shape[1], train_X.shape[2])))
# model.add(Dropout(0.2)) #Drops 20% of layer to prevent overfitting

# model.add(LSTM(units = 20, return_sequences = False))
# model.add(Dropout(0.2))

model.add(Dense(units = 1))

# Scheduled learning_rate (learning rate slows down over epochs)
# Lower learning rate
adam = optimizers.Adam(learning_rate = .0022)
model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics=[r_squared])
# Need to put in later: callbacks = [EarlyStopping(monitor='loss', patience=10)],
# Try model.fit and get history
history = model.fit_generator(data_gen_train, epochs = 200, validation_data = data_gen_test)
score = model.evaluate_generator(data_gen_test, verbose=0)

predicted_stock_price = model.predict(test_X)
predicted_stock_price = min_max_scaler.inverse_transform(predicted_stock_price)

# MSE: 0.012109583243727684, R-squared: 0.9007326364517212 after 200 epochs, learning_rate = 0.0022, and 160 batch size, step size = 14
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
