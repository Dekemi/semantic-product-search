import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def semantic_search(query_embedding, index, products, k=3):
    distances, indices = index.search(
        np.array([query_embedding]), k
    )
    return products.iloc[indices[0]]

def keyword_search(query, products, k=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(products["description"])
    query_vector = vectorizer.transform([query])
    scores = (tfidf_matrix @ query_vector.T).toarray().flatten()
    top_indices = scores.argsort()[::-1][:k]
    return products.iloc[top_indices]
