from fastapi import FastAPI
from pydantic import BaseModel
from src.recommender import recommend

app = FastAPI()

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "SHL Recommendation API is running"}

@app.post("/recommend")
def get_recommendations(query: Query):
    results = recommend(query.text)
    return {"recommendations": results}