import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

def clean_query(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def recommend(query, df, model, corpus_embeddings,
              tfidf_vectorizer, tfidf_matrix,
              alpha=0.7, top_n=5):

    cleaned_query = clean_query(query)

    # Semantic similarity
    query_embedding = model.encode([cleaned_query], convert_to_numpy=True)[0]
    semantic_scores = np.dot(corpus_embeddings, query_embedding) / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Lexical similarity
    query_tfidf = tfidf_vectorizer.transform([cleaned_query])
    lexical_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Hybrid score
    final_scores = alpha * semantic_scores + (1 - alpha) * lexical_scores

    df["similarity_score"] = final_scores

    results = df.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    return results[["title", "similarity_score", "url"]]