import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask
# app = Flask(__name__)
vectorizer = TfidfVectorizer(smooth_idf=True, norm='l2')

# Create a list of documents

# pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
# vec=pickle.load(open("vectorizer.pickle", 'rb'))


def vectorize_dic(Final):
    # documents = list(Final)
    # vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    doc_vectors = vectorizer.fit_transform(Final)
    return doc_vectors


def vectorize_query(processed_query):
    query_vector = vectorizer.transform([processed_query])
    return query_vector
