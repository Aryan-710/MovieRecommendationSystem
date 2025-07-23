import streamlit as st
import pickle

# Load the model and data
@st.cache_resource
def load_model():
    with open('movie_recommendation.pkl', 'rb') as f:
        knn, movies, csr_data = pickle.load(f)
    return knn, movies.reset_index(drop=True), csr_data

knn, movies, csr_data = load_model()

# Map titles to row numbers (length matches csr_data)
# ASSUMPTION: First N rows of `movies` are used in model
title_to_index = {
    title.lower(): idx for idx, title in enumerate(movies['title'][:csr_data.shape[0]].fillna(''))
}

# Recommend function
def recommend(movie_title):
    movie_title = movie_title.lower().strip()

    if movie_title not in title_to_index:
        return ["❌ Movie not found or not supported for recommendation."]

    row_num = title_to_index[movie_title]
    distances, indices = knn.kneighbors(csr_data.getrow(row_num), n_neighbors=6)
    recommended_indices = indices[0][1:]
    return movies.iloc[recommended_indices]['title'].tolist()

# Streamlit app UI
st.title("🎬 Movie Recommender System")

titles = sorted(movies['title'].dropna().astype(str).tolist())
movie_input = st.selectbox("Search or select a movie:", titles)

if st.button("Recommend"):
    recommendations = recommend(movie_input)
    st.subheader("🎥 Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)

