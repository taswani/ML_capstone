import data_pipeline as dp

result_df = dp.prepare_data(dp.price_csv, dp.headline_csv)
print(result_df)
