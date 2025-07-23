import streamlit as st
import pickle

# Load model and data
@st.cache_resource
def load_model():
    with open('movie_recommender.pkl', 'rb') as f:
        knn, movies, csr_data = pickle.load(f)  # ✅ Unpack all 3
    return knn, movies, csr_data

knn, movies, csr_data = load_model()

# Recommendation function
def recommend(movie_title):
    movie_title = movie_title.lower()
    if movie_title not in movies['title'].str.lower().values:
        return ["❌ Movie not found. Try a different title."]

    index = movies[movies['title'].str.lower() == movie_title].index[0]
    distances, indices = knn.kneighbors(csr_data[index], n_neighbors=6)
    recommended_indices = indices[0][1:]  # skip input movie itself
    return movies.iloc[recommended_indices]['title'].tolist()

# UI
st.title("🎬 Movie Recommender System")

titles = sorted(movies['title'].dropna().astype(str).tolist())
movie_input = st.selectbox("🔍 Search or select a movie:", titles)

if st.button("Recommend"):
    recommendations = recommend(movie_input)
    st.subheader("🎥 Recommended Movies:")
    for movie in recommendations:
        st.write("👉", movie)
