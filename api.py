from fastapi import FastAPI
from pydantic import BaseModel
from src.data_loader import load_data
from src.model import build_embedding_model, encode_corpus, build_tfidf
from src.recommender import recommend

app = FastAPI(title="SHL Recommendation API")

# Load everything once at startup
df = load_data()
model = build_embedding_model()
corpus_embeddings = encode_corpus(model, df["clean_text"].tolist())
tfidf_vectorizer, tfidf_matrix = build_tfidf(df["clean_text"])

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def get_recommendations(request: QueryRequest):

    results = recommend(
        request.query,
        df,
        model,
        corpus_embeddings,
        tfidf_vectorizer,
        tfidf_matrix
    )

    output = []

    for _, row in results.iterrows():
        output.append({
            "title": row["title"],
            "url": row["url"],
            "score": float(row["hybrid_score"])
        })

    return {"recommendations": output}