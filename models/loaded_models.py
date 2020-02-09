from joblib import load
import numpy as np
import pandas as pd
from keras.models import load_model
from data import DataPreparation, Query, result_df, r_squared

# load classical machine learning model
classical_model = load('classical_model.joblib')
# load feed-forward neural network from keras with custom r-squared metric function
ff_model = load_model('ff_model.h5', custom_objects={'r_squared': r_squared})
# Importing lstm model in case for later testing
lstm_model = load_model('lstm_model.h5', custom_objects={'r_squared': r_squared})

# TODO: Query with data to test model prediction
# Use this as test data:
# 1828.949951,1831.089966,1802.219971, "Russian officials are investigating Apple’s moves to remove parental control apps from its App Store shortly after it released a competing service.// Amazon is trying, starting with a new Jim Gaffigan set. So is HBO, though it’s more niche with acts like Julio Torres. And don’t count out YouTube.// “David Bowie: Finding Fame” airs on Showtime. And stream a Kurt Weill opera on BroadwayHD.// Toni Morrison dies. Nicolas Cage. Five years after Ferguson. What makes an American? A history of green lawns. Woodstock at 50. And more."

dp = DataPreparation(result_df)

X_train, X_test, y_train, y_test = dp.time_series_split(n=5)
min_max_scaler = dp.min_max_scaling(X_train, X_test, y_train, y_test, True)

q = Query(1828.949951,
            1831.089966,
            1802.219971,
            "Russian officials are investigating Apple’s moves to remove parental control apps from its App Store shortly after it released a competing service.// Amazon is trying, starting with a new Jim Gaffigan set. So is HBO, though it’s more niche with acts like Julio Torres. And don’t count out YouTube.// “David Bowie: Finding Fame” airs on Showtime. And stream a Kurt Weill opera on BroadwayHD.// Toni Morrison dies. Nicolas Cage. Five years after Ferguson. What makes an American? A history of green lawns. Woodstock at 50. And more.",
            result_df)

# Min-max scaling queries for model consumption, with reshaping to fit the min_max_scaler
converted_query = q.convert_data()
converted_query = np.array(converted_query).reshape(1, -1)
scaled_query = min_max_scaler.transform(converted_query)

# Predictions across different models, with reshaping to fit the min_max_scaler
classical_prediction = classical_model.predict(scaled_query).reshape(1, -1)
classical_prediction = min_max_scaler.inverse_transform(classical_prediction)

ff_prediction = ff_model.predict(scaled_query).reshape(1, -1)
ff_prediction = min_max_scaler.inverse_transform(ff_prediction)

# Print statement for checking values
print('Classical model prediction: {}, Feed-Forward model prediction: {}'.format(classical_prediction[0][0], ff_prediction[0][0]))
