import data_pipeline as dp
from dask import dataframe as dd
from text_classification import *

result_df = sentiment_analysis(result_df, processed_features)

# Needed in order to confirm changes made to data via pipeline
# print(result_df)

# TODO: Convert data to Dask dataframe, running into issue where you can't index a dask dataframe
result_dd = dd.from_pandas(result_df, npartitions=1)
# hyper parameter tuning using dask as well
