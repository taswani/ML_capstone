import data_pipeline as dp

corpus = dp.vectorization(dp.prepare_data(dp.price_csv, dp.headline_csv))
print(corpus)
