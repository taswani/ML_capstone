import pandas as pd

amazon_df = pd.read_csv("combined_amazon_date_data.csv", index_col=False)

# this is an effort to remove dates that are non-sensical from the data
amazon_df['Date'] = pd.to_datetime(amazon_df['Date'], errors='coerce')
amazon_df = amazon_df.dropna(subset=['Date'])

amzn_df = pd.read_csv("AMZN.csv", index_col=False)

# this is an effort to remove dates that are non-sensical from the data
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'], errors='coerce')
amzn_df = amzn_df.dropna(subset=['Date'])

# merge df on date
result = pd.merge(amazon_df, amzn_df, how="left",  on='Date')

# pushing df to a csv
result.to_csv("final.csv")
