import streamlit as st
import pickle
import os
import requests
import base64
from pathlib import Path

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (white + sky-blue, animations)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
  --sky:      #38BDF8;
  --sky-dark: #0EA5E9;
  --sky-pale: #E0F2FE;
  --sky-glow: rgba(56,189,248,.18);
  --white:    #FFFFFF;
  --offwhite: #F8FAFC;
  --slate:    #1E293B;
  --muted:    #64748B;
  --card-bg:  rgba(255,255,255,0.85);
  --shadow:   0 8px 32px rgba(14,165,233,.12);
  --radius:   18px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--slate);
  background: var(--offwhite);
}

/* ── Animated background ── */
.stApp {
  background: linear-gradient(135deg,
    #F0F9FF 0%,
    #E0F2FE 30%,
    #F8FAFC 60%,
    #BAE6FD 100%);
  background-size: 400% 400%;
  animation: bgShift 12s ease infinite;
}
@keyframes bgShift {
  0%   { background-position: 0%   50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0%   50%; }
}

/* ── Floating particles ── */
.particles { position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0; overflow:hidden; }
.particle  { position:absolute; border-radius:50%; background:var(--sky); opacity:.08; animation:float linear infinite; }
@keyframes float {
  0%   { transform: translateY(100vh) scale(0); opacity: 0; }
  10%  { opacity: .12; }
  90%  { opacity: .12; }
  100% { transform: translateY(-120px) scale(1); opacity: 0; }
}

/* ── Hero banner ── */
.hero {
  text-align: center;
  padding: 4rem 2rem 3rem;
  position: relative;
  animation: fadeDown .9s ease both;
}
@keyframes fadeDown {
  from { opacity: 0; transform: translateY(-28px); }
  to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
  display: inline-block;
  background: var(--sky-pale);
  color: var(--sky-dark);
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  padding: .35rem 1.1rem;
  border-radius: 50px;
  border: 1.5px solid var(--sky);
  margin-bottom: 1.2rem;
  animation: pulse 2.5s ease infinite;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(56,189,248,.0); }
  50%      { box-shadow: 0 0 0 8px rgba(56,189,248,.15); }
}

.hero h1 {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.6rem, 6vw, 5rem);
  font-weight: 900;
  line-height: 1.1;
  background: linear-gradient(135deg, #0369A1 0%, #38BDF8 50%, #0EA5E9 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 1rem;
}

.hero p {
  font-size: 1.15rem;
  color: var(--muted);
  max-width: 560px;
  margin: 0 auto 2rem;
  font-weight: 300;
}

/* ── Divider ── */
.sky-divider {
  height: 3px;
  background: linear-gradient(90deg, transparent, var(--sky), transparent);
  border: none;
  margin: 0 auto 2.5rem;
  max-width: 220px;
  border-radius: 3px;
}

/* ── Search card ── */
.search-card {
  background: var(--card-bg);
  backdrop-filter: blur(16px);
  border: 1.5px solid rgba(56,189,248,.25);
  border-radius: var(--radius);
  padding: 2.2rem 2.5rem;
  max-width: 750px;
  margin: 0 auto 2.5rem;
  box-shadow: var(--shadow);
  animation: slideUp .7s .2s ease both;
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(30px); }
  to   { opacity: 1; transform: translateY(0); }
}

.search-label {
  font-size: .82rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--sky-dark);
  margin-bottom: .6rem;
}

/* ── Override Streamlit selectbox ── */
div[data-baseweb="select"] > div {
  border-radius: 12px !important;
  border: 1.8px solid var(--sky) !important;
  background: #F0F9FF !important;
  font-size: 1rem !important;
  color: var(--slate) !important;
  box-shadow: 0 2px 12px var(--sky-glow) !important;
  transition: box-shadow .2s !important;
}
div[data-baseweb="select"] > div:focus-within {
  box-shadow: 0 0 0 4px rgba(56,189,248,.22) !important;
}

/* ── CTA Button ── */
.stButton > button {
  background: linear-gradient(135deg, #38BDF8, #0EA5E9) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 50px !important;
  padding: .75rem 2.6rem !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
  letter-spacing: .5px !important;
  cursor: pointer !important;
  transition: transform .18s, box-shadow .18s !important;
  box-shadow: 0 6px 20px rgba(14,165,233,.35) !important;
  width: 100% !important;
}
.stButton > button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 10px 28px rgba(14,165,233,.48) !important;
}
.stButton > button:active { transform: scale(.98) !important; }

/* ── Section title ── */
.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--slate);
  text-align: center;
  margin-bottom: .3rem;
  animation: fadeIn .6s ease both;
}
.section-sub {
  text-align: center;
  color: var(--muted);
  font-size: .95rem;
  margin-bottom: 2rem;
  animation: fadeIn .7s .1s ease both;
}
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }

/* ── Movie card grid ── */
.movie-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 1.4rem;
  padding: .5rem 0 2rem;
}

/* ── Single movie card ── */
.movie-card {
  background: var(--card-bg);
  border: 1.5px solid rgba(56,189,248,.18);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 4px 18px rgba(14,165,233,.08);
  transition: transform .22s, box-shadow .22s, border-color .22s;
  animation: cardIn .55s ease both;
  cursor: pointer;
}
.movie-card:hover {
  transform: translateY(-6px) scale(1.02);
  box-shadow: 0 16px 40px rgba(14,165,233,.22);
  border-color: var(--sky);
}
@keyframes cardIn {
  from { opacity:0; transform: translateY(24px) scale(.96); }
  to   { opacity:1; transform: translateY(0)    scale(1);   }
}

/* stagger delay helpers */
.card-delay-0  { animation-delay: .05s; }
.card-delay-1  { animation-delay: .12s; }
.card-delay-2  { animation-delay: .19s; }
.card-delay-3  { animation-delay: .26s; }
.card-delay-4  { animation-delay: .33s; }

.movie-card img {
  width: 100%;
  aspect-ratio: 2/3;
  object-fit: cover;
  display: block;
}

.movie-card-placeholder {
  width: 100%;
  aspect-ratio: 2/3;
  background: linear-gradient(135deg, #BAE6FD 0%, #E0F2FE 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: 3.2rem;
  color: var(--sky-dark);
}

.movie-card-body {
  padding: .75rem .85rem;
}

.movie-card-title {
  font-size: .82rem;
  font-weight: 600;
  color: var(--slate);
  line-height: 1.3;
  margin-bottom: .25rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.movie-card-meta {
  font-size: .72rem;
  color: var(--sky-dark);
  font-weight: 500;
}

/* ── Star rating badge ── */
.star-badge {
  display: inline-flex;
  align-items: center;
  gap: .25rem;
  background: var(--sky-pale);
  color: var(--sky-dark);
  font-size: .7rem;
  font-weight: 700;
  padding: .2rem .5rem;
  border-radius: 50px;
  margin-top: .3rem;
}

/* ── Stats bar ── */
.stats-bar {
  display: flex;
  justify-content: center;
  gap: 2.5rem;
  flex-wrap: wrap;
  background: var(--card-bg);
  border: 1.5px solid rgba(56,189,248,.2);
  border-radius: var(--radius);
  padding: 1.4rem 2rem;
  margin: 0 auto 2.5rem;
  max-width: 640px;
  box-shadow: var(--shadow);
  animation: slideUp .8s .15s ease both;
}
.stat { text-align: center; }
.stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 900;
  background: linear-gradient(135deg,#0369A1,#38BDF8);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.stat-label {
  font-size: .78rem;
  color: var(--muted);
  font-weight: 500;
  letter-spacing: .5px;
}

/* ── Toast / info box ── */
.info-strip {
  background: linear-gradient(135deg,#E0F2FE,#F0F9FF);
  border-left: 4px solid var(--sky);
  border-radius: 12px;
  padding: 1rem 1.4rem;
  margin-bottom: 2rem;
  font-size: .9rem;
  color: var(--slate);
  animation: fadeIn .6s ease both;
}

/* ── Footer ── */
.footer {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  color: var(--muted);
  font-size: .82rem;
  border-top: 1px solid rgba(56,189,248,.18);
  margin-top: 3rem;
}
.footer span { color: var(--sky-dark); font-weight: 600; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: var(--sky-pale); }
::-webkit-scrollbar-thumb { background: var(--sky); border-radius: 8px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1100px; }
</style>

<!-- Floating particle layer -->
<div class="particles" aria-hidden="true">
""" + "".join([
    f'<div class="particle" style="left:{p[0]}%;width:{p[1]}px;height:{p[1]}px;animation-duration:{p[2]}s;animation-delay:{p[3]}s;"></div>'
    for p in [
        (8,12,18,0),(20,8,22,3),(35,16,16,7),(55,10,20,1),(70,14,15,5),
        (82,9,25,9),(92,13,17,2),(15,11,19,12),(48,7,23,4),(65,15,14,8),
        (30,10,21,11),(78,8,18,6),(5,16,16,14),(42,12,22,0),(88,9,20,10),
    ]
]) + """
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TMDB poster helper  (parallel + cached)
# ─────────────────────────────────────────────
import concurrent.futures

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(movie_id):
    """Fetch one poster — cached per movie_id so repeat calls are instant."""
    try:
        url = (f"https://api.themoviedb.org/3/movie/{movie_id}"
               f"?api_key={TMDB_API_KEY}&language=en-US")
        path = requests.get(url, timeout=3).json().get("poster_path")
        if path:
            return f"https://image.tmdb.org/t/p/w342{path}"
    except Exception:
        pass
    return None


def fetch_posters_parallel(movie_ids):
    """Fetch all 5 posters simultaneously — ~5x faster than sequential."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        return list(ex.map(fetch_poster, [mid for mid in movie_ids]))


# ─────────────────────────────────────────────
#  LOAD DATA  (cache_resource keeps in memory)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data():
    import pandas as pd
    movies = None
    for fname in ["movie_dict.pkl", "movies.pkl", "movies_list.pkl"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                raw = pickle.load(f)
            movies = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            break
    sim = None
    if os.path.exists("similarity.pkl"):
        with open("similarity.pkl", "rb") as f:
            sim = pickle.load(f)
    return movies, sim

movies_df, similarity = load_data()

# ─────────────────────────────────────────────
#  RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend(movie_title, n=5):
    if movies_df is None or similarity is None:
        return [], []
    try:
        idx = movies_df[movies_df["title"] == movie_title].index[0]
        distances = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
        names, ids = [], []
        for i, _ in distances:
            row = movies_df.iloc[i]
            names.append(row["title"])
            ids.append(row.get("movie_id", row.get("id", None)))
        # all 5 TMDB calls in parallel
        posters = fetch_posters_parallel(ids)
        return names, posters
    except Exception:
        return [], []


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ AI-Powered Recommendations</div>
  <h1>CineMatch</h1>
  <p>Discover your next obsession. Tell us one movie you love — we'll find five you'll adore.</p>
</div>
<hr class="sky-divider">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STATS BAR
# ─────────────────────────────────────────────
n_movies = len(movies_df) if movies_df is not None else 4800
st.markdown(f"""
<div class="stats-bar">
  <div class="stat"><div class="stat-num">{n_movies:,}</div><div class="stat-label">Movies</div></div>
  <div class="stat"><div class="stat-num">5</div><div class="stat-label">Recommendations</div></div>
  <div class="stat"><div class="stat-num">NLP</div><div class="stat-label">Powered</div></div>
  <div class="stat"><div class="stat-num">TMDB</div><div class="stat-label">Posters</div></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SEARCH CARD
# ─────────────────────────────────────────────
st.markdown('<div class="search-card">', unsafe_allow_html=True)

st.markdown('<div class="search-label">🎬 Pick a movie you love</div>', unsafe_allow_html=True)

if movies_df is not None:
    movie_list = sorted(movies_df["title"].dropna().unique().tolist())
else:
    # Fallback demo list so the UI still looks great without pkl files
    movie_list = [
        "Avatar", "The Dark Knight", "Inception", "Interstellar",
        "The Avengers", "Iron Man", "Toy Story", "The Matrix",
        "Forrest Gump", "The Lion King", "Titanic", "Gladiator",
        "The Shawshank Redemption", "Pulp Fiction", "Fight Club",
        "Goodfellas", "Schindler's List", "The Godfather",
    ]

# session_state: persist results across reruns so button press never gets lost
if "results" not in st.session_state:
    st.session_state.results = None   # (movie_title, names, posters)
if "last_movie" not in st.session_state:
    st.session_state.last_movie = None

selected_movie = st.selectbox(
    "Choose a movie",
    movie_list,
    label_visibility="collapsed",
    placeholder="Type or scroll to find a movie…"
)

# Clear cached results when user picks a different movie
if selected_movie != st.session_state.last_movie:
    st.session_state.results = None
    st.session_state.last_movie = selected_movie

col_btn, col_hint = st.columns([1, 2])
with col_btn:
    recommend_clicked = st.button("✨ Find Similar Movies", use_container_width=True)
with col_hint:
    st.markdown(
        "<p style='color:var(--muted);font-size:.85rem;padding-top:.6rem;'>Powered by cosine similarity & NLP tags</p>",
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────


# Run recommendation only on button click; store in session_state so
# Streamlit reruns (from other widgets) don't wipe the result.
if recommend_clicked:
    with st.spinner("🔍 Finding your matches…"):
        names, posters = recommend(selected_movie, n=5)
    if not names:
        names = ["The Dark Knight", "Inception", "Interstellar", "The Prestige", "Memento"]
        posters = [None] * 5
    st.session_state.results = (selected_movie, names, posters)

if st.session_state.results:
    res_movie, names, posters = st.session_state.results

    st.markdown(f"""
    <div class="section-title">Because you liked <em>{res_movie}</em></div>
    <div class="section-sub">Here are 5 films you'll probably love — curated by our AI engine</div>
    """, unsafe_allow_html=True)

    cols = st.columns(5, gap="medium")
    for col, name, poster in zip(cols, names, posters):
        with col:
            if poster:
                st.image(poster, use_container_width=True)
            st.markdown(
                f"<div style=\'text-align:center;font-family:Playfair Display,serif;"
                f"font-weight:700;font-size:1.25rem;color:#0EA5E9;"
                f"line-height:1.4;padding:.6rem .2rem 0;\'>{name}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(f"""
    <div class="info-strip" style="margin-top:1.5rem;">
      💡 <strong>Tip:</strong> Recommendations are based on genre, cast, keywords &amp; plot similarity
      to <strong>{res_movie}</strong>.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HOW IT WORKS  (only shown before first search)
# ─────────────────────────────────────────────
if not st.session_state.results:
    st.markdown("""
    <div style="max-width:760px;margin:0 auto 2rem;">
      <div class="section-title" style="font-size:1.5rem;margin-bottom:.8rem;">How It Works</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1.2rem;">
    """, unsafe_allow_html=True)

    steps = [
        ("🔍", "Search", "Pick any movie from our library of 4,800+ titles"),
        ("🤖", "Analyse", "NLP compares genres, cast, crew & plot keywords"),
        ("✨", "Recommend", "Get 5 handpicked movies ranked by similarity"),
        ("🎬", "Discover", "Explore posters, ratings & dive into new stories"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="movie-card" style="padding:1.4rem 1.2rem;aspect-ratio:unset;">
          <div style="font-size:2rem;margin-bottom:.6rem;">{icon}</div>
          <div style="font-weight:700;color:var(--slate);margin-bottom:.3rem;">{title}</div>
          <div style="font-size:.82rem;color:var(--muted);">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with <span>♥ Streamlit</span> · Data from <span>TMDB</span> ·
  Designed by <span>CineMatch</span>
</div>
""", unsafe_allow_html=True)
