<div align="center">

# 🎬 CineMatch — Movie Recommendation System

<img src="screenshot.png" alt="CineMatch App Screenshot" width="100%" style="border-radius:12px;"/>

<br/>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommend-system-3nhywrira9utd9yyjlsw6f.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB-API-01B4E4?style=flat-square&logo=themoviedatabase&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-38BDF8?style=flat-square)

**An AI-powered content-based movie recommender built with Python & Streamlit.**  
Pick any movie from 4,800+ titles and instantly get 5 personalized recommendations — complete with live TMDB posters.

[🚀 Live Demo](https://movie-recommend-system-3nhywrira9utd9yyjlsw6f.streamlit.app/) • [📖 How It Works](#-how-it-works) • [⚙️ Setup](#-local-setup) • [🧠 Tech Stack](#-tech-stack)

</div>

---

## ✨ Why This Project?

Finding a great movie to watch next is harder than it should be. Streaming platforms bury recommendations behind algorithms you can't control, and most "similar movie" lists are just studio cash-grabs.

**CineMatch solves this by:**
- Using **NLP + cosine similarity** on real movie metadata (genres, cast, crew, keywords, overview) — not just ratings or popularity
- Giving you **transparent recommendations** based on content DNA, not what the platform wants you to watch
- Being completely **open-source and ad-free**, running on a clean, fast UI

This project was built to learn and demonstrate **end-to-end ML deployment** — from raw data preprocessing in a Jupyter notebook all the way to a live, public web app.

---

## 🖼️ App Preview

| Search & Recommend | Live Results |
|---|---|
| Pick any of 4,800+ movies from the dropdown | Get 5 similar movies with TMDB posters instantly |

> **Try it live →** [movie-recommend-system-3nhywrira9utd9yyjlsw6f.streamlit.app](https://movie-recommend-system-3nhywrira9utd9yyjlsw6f.streamlit.app/)

---

## 🧠 How It Works

### Step 1 — Data Collection
The dataset is the **TMDB 5000 Movie Dataset** from Kaggle, containing metadata for 4,803 movies including genres, cast, crew, keywords, and plot overview.

### Step 2 — Feature Engineering (NLP Tags)
For each movie, we extract and combine:
- **Genres** → `["Action", "Adventure"]`
- **Top 3 Cast members** → `["Leonardo DiCaprio", "Tom Hardy"]`
- **Director** → `["Christopher Nolan"]`
- **Keywords** → `["dream", "heist", "subconscious"]`
- **Overview** → stemmed bag-of-words from the plot

All of these are merged into a single **"tags" string** per movie:

```
inception action adventure thriller leonardodicaprio tomhardy josephgordonlevitt
christophernolan dream heist subconscious mind bending
```

### Step 3 — Vectorisation
We apply **CountVectorizer** (5,000 features, English stop-words removed) to convert all tags into numerical vectors.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
```

### Step 4 — Cosine Similarity
We compute a **4803 × 4803 similarity matrix** — every movie scored against every other.

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

> **Why cosine similarity?** It measures the *angle* between two vectors, not their magnitude — so a short movie with 3 keywords and a long one with 30 keywords are compared fairly on what they share, not how much they have.

### Step 5 — Recommendation
```python
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    return [movies.iloc[i[0]].title for i in distances[1:6]]
```

### Step 6 — Live Posters via TMDB API
Each recommended movie's `movie_id` is used to fetch a live poster from The Movie Database API, displayed using `st.image()`.

---

## 🏗️ Project Structure

```
Movie-Recommend-System/
│
├── app.py                          ← Streamlit web app (main entry point)
├── movie-recommender-system.ipynb  ← Data preprocessing + model training notebook
│
├── movie_dict.pkl                  ← Serialised movie DataFrame (title, movie_id, tags)
├── similarity.pkl                  ← Precomputed 4803×4803 cosine similarity matrix
│
├── .streamlit/
│   └── config.toml                 ← Sky-blue UI theme configuration
│
├── screenshot.png                  ← App preview image
├── requirements.txt                ← Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/parthTyagi-tech/Movie-Recommend-System.git
cd Movie-Recommend-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

> **Note:** The `movie_dict.pkl` and `similarity.pkl` files must be present in the root directory. Run the Jupyter notebook first if they are missing.

---

## 🧪 Retrain the Model

```bash
pip install jupyter
jupyter notebook movie-recommender-system.ipynb
```

Run all cells — this regenerates `movie_dict.pkl` and `similarity.pkl` from the raw TMDB dataset.

---

## 🧰 Tech Stack

| Layer | Tool | Why We Used It |
|---|---|---|
| **Language** | Python 3.8+ | Industry standard for ML/data science |
| **Web Framework** | Streamlit | Fastest way to turn a Python script into a live web app — no HTML/JS needed |
| **ML / NLP** | scikit-learn | CountVectorizer + cosine_similarity — lightweight, no GPU needed |
| **Data** | Pandas, NumPy | Efficient DataFrame manipulation for preprocessing |
| **Serialisation** | Pickle | Save/load trained vectors and similarity matrix instantly |
| **Poster API** | TMDB API | Free, reliable movie metadata and high-quality poster images |
| **Deployment** | Streamlit Community Cloud | Free hosting, auto-deploys from GitHub on every push |
| **Stemming** | NLTK PorterStemmer | Reduces words to root form — "running" = "run" = better tag matching |

---

## 🎨 UI Design

| Element | Choice | Reason |
|---|---|---|
| Color palette | White + Sky Blue `#38BDF8` | Clean, cinematic, easy on the eyes |
| Background | Animated CSS gradient | Adds life without distracting from content |
| Typography | Playfair Display + DM Sans | Serif for elegance, sans-serif for readability |
| Animations | CSS keyframes (fadeDown, slideUp, cardIn) | Smooth, performant — no JS libraries needed |
| Layout | `st.columns(5)` | Native Streamlit grid, responsive and reliable |

---

## 📦 Requirements

```
streamlit>=1.28.0
requests>=2.28.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.24.0
nltk>=3.7
Pillow>=9.0.0
```

---

## 🔮 Future Improvements

- [ ] Add **collaborative filtering** (user-based recommendations)
- [ ] Include **movie ratings & release year** in the cards
- [ ] Add a **"Surprise Me"** button for random picks
- [ ] Support **multi-movie input** ("I liked X and Y, recommend Z")
- [ ] Add **genre filter** sidebar
- [ ] Switch to **TF-IDF** vectorization for better weighting of rare keywords

---

## 🙌 Acknowledgements

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — Kaggle
- [The Movie Database (TMDB)](https://www.themoviedb.org/) — Poster images via API
- [Streamlit](https://streamlit.io/) — For making deployment this simple
- [scikit-learn](https://scikit-learn.org/) — ML utilities

---

<div align="center">

Made with ❤️ by **Parth Tyagi**
![screenshot](https://github.com/user-attachments/assets/9813f57b-e381-450f-8a81-073b82bf75b7)

⭐ **Star this repo if you found it useful!**

</div>
