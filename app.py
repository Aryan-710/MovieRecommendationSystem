import streamlit as st
import pickle

# Load model and data
@st.cache_resource
def load_model():
    with open('movie_recommendation.pkl', 'rb') as f:
        knn, movies, csr_data = pickle.load(f)
    return knn, movies, csr_data

knn, movies, csr_data = load_model()

# Recommendation logic using KNN
def recommend(movie_title):
    movie_title = movie_title.lower()

    # Find the row number (not DataFrame index!)
    matched = movies[movies['title'].str.lower() == movie_title]
    if matched.empty:
        return ["❌ Movie not found. Try a different title."]
    
    row_num = matched.index[0]  # safe index from dataframe
    row_num = movies.reset_index().index[movies['title'].str.lower() == movie_title][0]  # force reset indexing

    distances, indices = knn.kneighbors(csr_data[row_num], n_neighbors=6)
    recommended_indices = indices[0][1:]  # skip input movie itself
    return movies.iloc[recommended_indices]['title'].tolist()

# App UI
st.title("🎬 Movie Recommender System")

titles = sorted(movies['title'].dropna().astype(str).tolist())
movie_input = st.selectbox("Search or select a movie:", titles)

if st.button("Recommend"):
    recommendations = recommend(movie_input)
    st.subheader("🎥 Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)
