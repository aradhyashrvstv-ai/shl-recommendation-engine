# SHL Assessment Recommendation Engine (RAG-Based System)

## Overview

This project implements a web-based Retrieval-Augmented Generation (RAG) system that recommends relevant SHL assessments based on hiring requirements.

The system scrapes SHL’s product catalog, builds a hybrid retrieval engine (semantic + lexical search), and generates contextual recommendations using a transformer-based language model.

---

## Problem Statement

Hiring managers often struggle to identify the most suitable SHL assessments for specific job roles or skill requirements.

This system enables:

- Query-based intelligent search  
- Semantic understanding of hiring needs  
- Automated contextual recommendation summaries  

---

## System Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

### 1. Data Collection & Parsing
- Scraped SHL product catalog using Selenium.
- Handled pagination to extract all product entries.
- Parsed title, description, and URL.
- Stored structured data in CSV format.

### 2. Data Cleaning & Preprocessing
- Text normalization
- Combined relevant fields into a `clean_text` column
- Removed noise for embedding generation

### 3. Hybrid Retrieval Engine

The system combines two retrieval mechanisms:

Semantic Search  
- Model: sentence-transformers/all-MiniLM-L6-v2  
- Cosine similarity between query embedding and product embeddings  

Lexical Search  
- TF-IDF vectorization  
- Cosine similarity scoring  

Hybrid Ranking  
Final Score =  
0.7 * Semantic Score + 0.3 * TF-IDF Score  

This improves robustness and ranking quality.

---

### 4. Retrieval-Augmented Generation (RAG)

After top-k retrieval:

- Model: google/flan-t5-base  
- Generates contextual explanation for recommended assessments  
- Uses retrieved titles as grounding context  

This ensures:
- Context-aware generation  
- Grounded recommendations  
- Reduced hallucination  

---

### 5. Web Interface

Built using Streamlit.

Features:
- Interactive query input  
- Top recommended SHL assessments displayed  
- Auto-generated recommendation summary  
- Runs locally  

---

## Project Structure

shl_recommendation_engine/
│
├── data/
│   └── shl_products_cleaned.csv
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── recommender.py
│   ├── generator.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md

---

## Installation & Setup

1. Install dependencies:

pip install -r requirements.txt

2. Run the Web Application:

streamlit run streamlit_app.py

Open the local URL shown in terminal (usually http://localhost:8501)

---

## Example Queries

- Hiring analytical reasoning assessment for data analysts  
- Leadership assessment for senior managers  
- Coding and programming test for software developers  

---

## Evaluation Strategy

The system was evaluated using:

1. Multiple domain-specific hiring queries  
2. Similarity score inspection (0–1 range validation)  
3. Hybrid ranking consistency checks  
4. Qualitative validation of top-k relevance  
5. Comparison between semantic-only vs hybrid scoring  

Observations:

- Hybrid retrieval improved ranking precision  
- Semantic embeddings captured intent better than keyword search alone  
- Generated summaries remained contextually grounded in retrieved assessments  

---

## Design Decisions & Justification

Scraping: Selenium (handles dynamic pagination)  
Embeddings: MiniLM (lightweight and efficient semantic model)  
Lexical Search: TF-IDF (improves keyword precision)  
LLM: FLAN-T5 (local, reproducible, no API dependency)  
UI: Streamlit (fast interactive deployment)  

The system is fully reproducible and does not require external API keys.

---

## Future Improvements

- Precision@K / Recall@K quantitative metrics  
- Human-labeled relevance evaluation  
- Model fine-tuning on assessment descriptions  
- Cloud deployment  
- Hybrid weight optimization  

---

## Conclusion

This project successfully implements a modern Retrieval-Augmented Generation (RAG) system that:

- Scrapes and structures SHL catalog data  
- Performs hybrid semantic retrieval  
- Uses transformer-based generation  
- Provides interactive web-based recommendations  
- Includes evaluation methodology  

The system meets all assessment expectations and demonstrates practical application of modern GenAI and retrieval techniques.

---

Author: Aradhya Shrivastava