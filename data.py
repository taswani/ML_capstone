import pandas as pd

amazon_df = pd.read_csv("combined_amazon_date_data.csv", index_col=False)
amazon_df['Date'] = pd.to_datetime(amazon_df['Date'], errors='coerce')
amazon_df = amazon_df.dropna(subset=['Date'])

amzn_df = pd.read_csv("AMZN.csv", index_col=False)
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'], errors='coerce')
amzn_df = amzn_df.dropna(subset=['Date'])

result = pd.merge(amazon_df, amzn_df, how="left",  on='Date')

result.to_csv("final.csv")
