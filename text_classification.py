from data import result_df
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#In case you need data
# import nltk
# nltk.download('punkt')

#Creation of corpus from all the headlines
corpus = []
for line in result_df['Headlines']:
    corpus.append(line)

#TODO: preprocess text for vectorization


#Initializing Vectorizer
vectorizer = CountVectorizer()

#Transforming words to vectors
X = vectorizer.fit_transform(corpus)

corpus_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(corpus_df)
