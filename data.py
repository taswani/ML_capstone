import data_pipeline as dp
from text_classification import *

result_df = sentiment_analysis(result_df, processed_features)

# Needed in order to confirm changes made to data via pipeline
# print(result_df)

# TODO: Convert data to Dask.dd
# hyper parameter tuning using dask as well
