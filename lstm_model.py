from data import result_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate

X = result_df[['Open', 'High', 'Low', 'Average Mean', 'Polarity', 'Sentiment', 'Differential']]
y = result_df[['Close']]

# 1139 x 8
# print(result_df.shape)

# Time series specific train-test-split
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Scaling all values for a normalized input and output
min_max_scaler = MinMaxScaler()

X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
y_train = min_max_scaler.fit_transform(y_train)
y_test = min_max_scaler.fit_transform(y_test)

# Using keras' timeseriesgenerator in order to divide the data into batches
# Putting data into 3D for input to the LSTM
data_gen_train = TimeseriesGenerator(X_train, y_train,
                               length=60, sampling_rate=1,
                               batch_size=32)

data_gen_test = TimeseriesGenerator(X_test, y_test,
                               length=60, sampling_rate=1,
                               batch_size=32)

# Done for Input shape of LSTM
train_X, train_y = data_gen_train[0]
test_X, test_y = data_gen_test[0]

# Begin LSTM

model = Sequential()

# 50 neurons per layer till last one
model.add(LSTM(units = 50, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2)) #Drops 20% of layer to prevent overfitting

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
model.fit_generator(data_gen_train, epochs = 100, validation_data = data_gen_test)
score = model.evaluate_generator(data_gen_test, verbose=0)

predicted_stock_price = model.predict(test_X)
predicted_stock_price = min_max_scaler.inverse_transform(predicted_stock_price)

# [0.022621948271989822, 0.7829457521438599] after 100 epochs
print("Scalar Loss: ", score[0])
print("Accuracy: ", score[1])
