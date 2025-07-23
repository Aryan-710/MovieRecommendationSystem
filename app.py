import streamlit as st
import pickle

# Load the saved model and data
@st.cache_resource
def load_model():
    with open('movie_recommender.pkl', 'rb') as file:
        movies, similarity = pickle.load(file)
    return movies, similarity

movies, similarity = load_model()

# Recommend movies based on title
def recommend(movie_title):
    movie_title = movie_title.lower()

    # Find movie index
    if movie_title not in movies['title'].str.lower().values:
        return ["❌ Movie not found. Try a different title."]

    index = movies[movies['title'].str.lower() == movie_title].index[0]

    # Get similarity scores
    distances = list(enumerate(similarity[index]))

    # Sort and get top 5 similar
    recommended_indices = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = movies['title'].iloc[[i[0] for i in recommended_indices]].tolist()

    return recommended_movies

st.title("🎬 Movie Recommendation App")

movie_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_input.strip():
        recommendations = recommend(movie_input)
        st.subheader("🎥 Recommended Movies:")
        for rec in recommendations:
            st.write("👉", rec)
    else:
        st.warning("⚠️ Please enter a movie name.")
