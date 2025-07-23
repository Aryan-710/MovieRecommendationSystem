import streamlit as st
import pickle

# Load the saved KNN model and movies DataFrame
@st.cache_resource
def load_model():
    with open('movie_recommender.pkl', 'rb') as file:
        knn, movies = pickle.load(file)
    return knn, movies

knn, movies = load_model()

# Recommend using NearestNeighbors model
def recommend(movie_title):
    movie_title = movie_title.lower()

    if movie_title not in movies['title'].str.lower().values:
        return ["❌ Movie not found. Try a different title."]

    index = movies[movies['title'].str.lower() == movie_title].index[0]

    distances, indices = knn.kneighbors(knn._fit_X[index], n_neighbors=6)
    recommended_indices = indices[0][1:]  # Skip the input movie itself
    recommended_movies = movies.iloc[recommended_indices]['title'].tolist()

    return recommended_movies


st.title("🎬 Movie Recommendation App")

st.markdown("Choose how you'd like to select a movie:")

option = st.radio("Selection Mode", ["Type Movie Name", "Choose from List"], horizontal=True)

if option == "Type Movie Name":
    movie_input = st.text_input("Enter a movie name:")
else:
    movie_input = st.selectbox("Pick a movie:", sorted(movies['title'].tolist()))

if st.button("Recommend"):
    if movie_input.strip():
        recommendations = recommend(movie_input)
        st.subheader("🎥 Recommended Movies:")
        for rec in recommendations:
            st.write("👉", rec)
    else:
        st.warning("⚠️ Please enter or select a movie name.")
