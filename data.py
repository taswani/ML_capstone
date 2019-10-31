import data_pipeline as dp
from text_classification import *

result_df = sentiment_analysis(result_df, processed_features)
print(result_df)
