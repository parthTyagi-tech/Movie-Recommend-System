import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

@st.cache_resource
def load_or_build():
    # Load saved movies dataframe
    movies = pickle.load(open("movies.pkl", "rb"))

    # Build similarity matrix on server (since similarity.pkl is too large for GitHub)
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_or_build()

selected_movie = st.selectbox("Select a movie", movies["title"].values)

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("✅", movie)
