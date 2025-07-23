import streamlit as st
import pickle

@st.cache_resource
def load_model():
    with open('movie_recommendation.pkl', 'rb') as f:
        knn, movies, csr_data = pickle.load(f)
    return knn, movies, csr_data

knn, movies, csr_data = load_model()

def recommend(movie_title):
    movie_title = movie_title.lower()
    matched = movies[movies['title'].str.lower() == movie_title]
    
    if matched.empty:
        return ["❌ Movie not found. Try a different title."]
    
    row_num = matched.index[0]
    
    # ✅ FIX: get correct sparse row
    distances, indices = knn.kneighbors(csr_data.getrow(row_num), n_neighbors=6)
    recommended_indices = indices[0][1:]  # Skip the input movie itself
    return movies.iloc[recommended_indices]['title'].tolist()

st.title("🎬 Movie Recommender System")

titles = sorted(movies['title'].dropna().astype(str).tolist())
movie_input = st.selectbox("Search or select a movie:", titles)

if st.button("Recommend"):
    recommendations = recommend(movie_input)
    st.subheader("🎥 Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)
