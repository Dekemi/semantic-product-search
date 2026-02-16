import streamlit as st
import pandas as pd
from embeddings import embed_text
from search import build_faiss_index, semantic_search, keyword_search

st.title("Semantic Product Search")

products = pd.read_csv("data/products.csv")

texts = products["title"] + ". " + products["description"]

embeddings = embed_text(texts.tolist())
index = build_faiss_index(embeddings)

query = st.text_input("Type what you're looking for:")

search_type = st.radio(
    "Choose search method:",
    ["Keyword Search", "Semantic Search"]
)

if query:
    if search_type == "Semantic Search":
        query_embedding = embed_text([query])[0]
        results = semantic_search(
            query_embedding, index, products
        )
    else:
        results = keyword_search(query, products)

    st.subheader("Results")
    st.dataframe(results)
