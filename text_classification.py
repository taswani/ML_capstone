import data_pipeline as dp

corpus = dp.vectorization(dp.prepare_data(dp.price_csv, dp.headline_csv))
print(corpus)

# TODO: Sentiment analysis via transfer learning since I have unlabeled data at the moment.
# Look to use pre-trained model, aggregrate the predictions and append to result_df
