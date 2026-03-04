import streamlit as st
import numpy as np
from src.data_loader import load_data
from src.model import build_embedding_model, encode_corpus, build_tfidf
from src.recommender import recommend
from src.generator import build_generator, generate_response

st.title("SHL Assessment Recommendation Engine (RAG)")

df = load_data()

model = build_embedding_model()
corpus_embeddings = encode_corpus(model, df["clean_text"].tolist())
tfidf_vectorizer, tfidf_matrix = build_tfidf(df["clean_text"])
generator = build_generator()

query = st.text_input("Enter hiring requirement:")

if query:
    results = recommend(
        query,
        df,
        model,
        corpus_embeddings,
        tfidf_vectorizer,
        tfidf_matrix
    )

    st.subheader("Top Recommended Assessments")

    retrieved_text = ""

    for _, row in results.iterrows():
        st.write(f"**{row['title']}**")
        st.write(row["url"])
        retrieved_text += row["title"] + "\n"

    st.subheader("Generated Recommendation Summary")

    summary = generate_response(generator, query, retrieved_text)
    st.write(summary)