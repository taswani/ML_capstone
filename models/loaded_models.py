from joblib import load
from keras.models import load_model
from data import result_df, r_squared, DataPreparation

# load classical machine learning model
classical_model = load('classical_model.joblib')
# load neural networks from keras with custom r-squared metric function
ff_model = load_model('ff_model.h5', custom_objects={'r_squared': r_squared})
lstm_model = load_model('lstm_model.h5', custom_objects={'r_squared': r_squared})
