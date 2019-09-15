import pandas as pd
import numpy as np
import datetime as dt

#reading csvs into respective dataframes
amazon_df = pd.read_csv("combined_amazon_date_data.csv", index_col=False)
amzn_df = pd.read_csv("AMZN.csv", index_col=False)

# this is an effort to remove dates that are non-sensical from the data
amazon_df['Date'] = pd.to_datetime(amazon_df['Date'], errors='coerce')
amazon_df = amazon_df.dropna(subset=['Date'])

#reading second csv into a dataframe
amzn_df = pd.read_csv("AMZN.csv", index_col=False)

# this is an effort to remove dates that are non-sensical from the data
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'], errors='coerce')
amzn_df = amzn_df.dropna(subset=['Date'])

# merge df on date
result_df = pd.merge(amazon_df, amzn_df, how="left",  on='Date')

# dates being sorted for forward fill
result_df = result_df.sort_values(by="Date")

# cover null values in data
result_df = result_df.fillna(method='ffill')

# dropping any mention of anything other than the company amazon
# work in progress as I see other keywords
to_drop = ['rainforest', 'forest', 'Brazil', 'river', 'jungle', 'River', 'pilots', 'gangs', 'drugs', 'OkCupid', 'dating']
result_df = result_df[~result_df['Headlines'].str.contains('|'.join(to_drop))]

# TODO: combining duplicates by date
result_df = result_df.groupby(['Date', 'Close'])['Headlines'].apply('// '.join).reset_index()

# pushing df to a csv
result_df.to_csv("final_amazon.csv")
