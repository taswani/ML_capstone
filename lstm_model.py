from data import result_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate

rnn_df = result_df[['Open', 'High', 'Low', 'Average Mean', 'Polarity', 'Sentiment', 'Differential', 'Close']]

column_labels = ['Open', 'High', 'Low', 'Average Mean', 'Polarity', 'Sentiment', 'Differential', 'Close']

# 1139 x 8
# print(rnn_df.shape)

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(rnn_df):
    df_train, df_test = rnn_df.iloc[train_index], rnn_df.iloc[test_index]

df_values = df_train.loc[:, column_labels].values

min_max_scaler = MinMaxScaler()

train = min_max_scaler.fit_transform(df_values)
test = min_max_scaler.transform(df_test.loc[:, column_labels])

# Training data set
X_train = []
y_train = []

for i in range(60, 949):
    X_train.append(train[i-60:i, :-1:1])
    y_train.append(train[i, -1])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 7))

# Testing dataset
X_test = []
y_test = []

for i in range(60, 189):
    X_test.append(test[i-60:i, :-1:1])
    y_test.append(test[i, -1])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 7))

# Begin LSTM

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 7)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 5, batch_size = 32)
score, acc = model.evaluate(X_test, y_test, batch_size = 32)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = min_max_scaler.inverse_transform(predicted_stock_price)

print(predicted_stock_price)
print()
print("Score: ", score)
print()
print("Accuracy: ", acc)
