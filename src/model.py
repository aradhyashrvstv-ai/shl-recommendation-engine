from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def build_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def encode_corpus(model, corpus):
    return model.encode(corpus, convert_to_numpy=True)

def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix