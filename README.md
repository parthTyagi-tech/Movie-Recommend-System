# 🎬 Movie Recommendation System

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-ff4b4b?logo=streamlit)](https://movie-recommend-system-3nhywrira9utd9yyjlsw6f.streamlit.app)

🔗 **Live Application:**  
👉 [Click Here to Try the App](https://movie-recommend-system-3nhywrira9utd9yyjlsw6f.streamlit.app)

---

## 📌 Project Overview

The Movie Recommendation System is a content-based filtering application that suggests similar movies based on user-selected preferences.

The system analyzes movie metadata such as genres, keywords, cast, crew, and overview text, and uses Natural Language Processing (NLP) techniques to compute similarity between movies.

By applying vectorization techniques and cosine similarity, the model identifies movies that are most similar to the selected title and provides personalized recommendations.

---

## 🎯 Key Features

✔ Content-based movie recommendations  
✔ Cosine similarity matching  
✔ Interactive Streamlit interface  
✔ Clean and responsive UI  
✔ Real-time recommendation results  

---

## 🧠 How It Works

1. Movie metadata is preprocessed and combined into a single text feature.
2. The text data is vectorized using CountVectorizer.
3. Cosine similarity is computed between movie vectors.
4. When a user selects a movie, the system retrieves the most similar movies based on similarity score.
5. The top recommended movies are displayed in the web interface.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **CountVectorizer**
- **Cosine Similarity**

---

## 📂 Project Structure

Movie-Recommend-System
│
├── app.py
├── movies.pkl
├── similarity.pkl
├── requirements.txt
└── dataset files


---

## 💡 Future Improvements

- Add collaborative filtering
- Integrate TMDB API for posters and ratings
- Add user-based recommendations
- Implement hybrid recommendation system
- Add genre-based filtering options

---

## 👨‍💻 Author

**Parth Tyagi**  
B.Tech – Mathematics & Computing  
Machine Learning & AI Enthusiast  

---

## ⭐ If You Like This Project

Please consider giving this repository a ⭐ on GitHub!
