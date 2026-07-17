# Semantic Product Search

A semantic search application that demonstrates the difference between traditional keyword search and vector-based semantic search.

## Features

- 🔍 Semantic search using text embeddings and FAISS
- 📄 Keyword search using TF-IDF
- ⚡ Fast similarity retrieval
- 📊 Interactive Streamlit dashboard for comparing search methods

## Technologies

- Python
- Streamlit
- FAISS
- Pandas
- NumPy
- Scikit-learn

## How it Works

1. Product titles and descriptions are converted into vector embeddings.
2. FAISS indexes the embeddings for efficient similarity search.
3. User queries are converted into embeddings and matched against the index.
4. Users can compare semantic search with traditional keyword search to see the difference in retrieval quality.

## Live Demo

Available on Streamlit Community Cloud:https://semantic-search-project.streamlit.app
