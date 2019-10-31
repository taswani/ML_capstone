import data_pipeline as dp
from textblob import TextBlob

result_df = dp.prepare_data(dp.price_csv, dp.headline_csv)
processed_features = dp.vectorization(result_df)

# TODO: Sentiment analysis via transfer learning since I have unlabeled data at the moment.
# Look to use pre-trained model, aggregrate the predictions and append to result_df

def sentiment_analysis(result_df, processed_features):
    # List for taking in the sentiments associated with the processed features
    sentiments = []
    # Using a pre-trained model to analyze sentiment for each headline
    for feature in processed_features:
        sentence = TextBlob(feature)
        if sentence.sentiment.polarity > 0:
            sentiments.append(1)
        else:
            sentiments.append(0)
    # Adding a new column in the dataframe and returning it
    result_df['Sentiment'] = sentiments
    # pushing df to a csv
    result_df.to_csv("final_amazon.csv")
    return result_df
