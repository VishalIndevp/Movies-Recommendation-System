import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS Part 1: Fonts, base, background, container, hero ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@700,900&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #04080f; color: #e2eaf4; }
.block-container { max-width: 720px !important; padding: 0rem 1rem 3rem !important; position: relative; z-index: 1; }
@media (min-width: 480px) { .block-container { padding: 0rem 2rem 4rem !important; } }
.hero { text-align: center; margin-bottom: 2.2rem; padding: 0 .5rem; }
.hero-eyebrow { font-size: clamp(.6rem, 2.5vw, .72rem); font-weight: 500; letter-spacing: .26em; text-transform: uppercase; color: #4d8eff; margin-bottom: .5rem; }
.hero-title { font-family: 'Satoshi', 'DM Sans', sans-serif; font-size: clamp(2rem, 7vw, 4.5rem); font-weight: 900; line-height: 1.05; letter-spacing: -.02em; color: #f5f2e8; margin: 0 0 .8rem; word-break: keep-all; white-space: nowrap; overflow: visible; }
.hero-title span { color: #3b82f6; }
.hero-sub { font-size: clamp(.8rem, 3vw, .95rem); font-weight: 300; color: #4a6080; letter-spacing: .02em; }
.divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #1a3a6e, transparent); margin: 0 0 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── CSS Part 2: Controls ──────────────────────────────────────────────────────
st.markdown("""
<style>
.select-label { font-size: .68rem; font-weight: 500; letter-spacing: .2em; text-transform: uppercase; color: #2e4d7a; margin-bottom: .4rem; }
div[data-baseweb="select"] > div { background: #080f1e !important; border: 1px solid #1a2d50 !important; border-radius: 8px !important; color: #c8d8f0 !important; font-family: 'DM Sans', sans-serif !important; font-size: clamp(.85rem, 3.5vw, .95rem) !important; transition: border-color .2s; }
div[data-baseweb="select"] > div:hover { border-color: #3b82f6 !important; }
div[data-testid="stSlider"] > div > div > div { background: #3b82f6 !important; }
div[data-testid="stSlider"] [role="slider"] { background: #3b82f6 !important; border: 2px solid #04080f !important; box-shadow: 0 0 0 3px rgba(59,130,246,.35) !important; width: 22px !important; height: 22px !important; }
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span,
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p { color: #c8d8f0 !important; font-family: 'DM Sans', sans-serif !important; }
div[data-testid="stTickBar"] > div { color: #2e4d7a !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] output,
[data-testid="stSlider"] .st-emotion-cache-1dp5vir,
[data-testid="stSlider"] [aria-valuetext],
[data-testid="stSlider"] ~ div p,
[data-testid="stSlider"] + div,
[data-testid="stSlider"] + div p { color: #c8d8f0 !important; }
div[data-testid="stSlider"] * { color: #c8d8f0 !important; }
div.stButton > button { width: 100%; margin-top: 1.2rem; padding: .9rem 2rem; background: #1d4ed8 !important; color: #fff !important; font-family: 'Satoshi', 'DM Sans', sans-serif; font-weight: 700; font-size: clamp(1rem, 4vw, 1.1rem); letter-spacing: .08em; border: none !important; border-radius: 8px; cursor: pointer; transition: background .2s, transform .15s, box-shadow .2s; min-height: 52px; touch-action: manipulation; box-shadow: 0 4px 24px rgba(29,78,216,.35); }
div.stButton > button:hover { background: #2563eb !important; color: #fff !important; border: none !important; transform: translateY(-2px); box-shadow: 0 6px 30px rgba(37,99,235,.45); }
div.stButton > button:active, div.stButton > button:focus { background: #1e40af !important; color: #fff !important; border: none !important; outline: none !important; box-shadow: 0 4px 24px rgba(29,78,216,.35) !important; transform: translateY(0); }
div.stButton > button:focus:not(:active) { background: #1d4ed8 !important; color: #fff !important; border: none !important; outline: none !important; box-shadow: 0 4px 24px rgba(29,78,216,.35) !important; }
</style>
""", unsafe_allow_html=True)

# ── CSS Part 3: Results, cards, footer ───────────────────────────────────────
st.markdown("""
<style>
.section-label { font-size: .65rem; font-weight: 500; letter-spacing: .22em; text-transform: uppercase; color: #3b82f6; margin-bottom: .3rem; margin-top: 2rem; }
.results-header { font-family: 'Satoshi', 'DM Sans', sans-serif; font-weight: 700; font-size: clamp(1.2rem, 5vw, 1.5rem); letter-spacing: -.01em; color: #f5f2e8; margin: .4rem 0 1rem; display: flex; align-items: center; gap: .6rem; }
.results-header::after { content: ""; flex: 1; height: 1px; background: #1a2d50; }
.movie-card { display: flex; align-items: center; gap: .8rem; background: #080f1e; border: 1px solid #102040; border-radius: 10px; padding: .85rem 1rem; margin-bottom: .5rem; transition: border-color .2s, transform .2s, box-shadow .2s; animation: slideIn .35s ease both; -webkit-tap-highlight-color: transparent; }
@media (min-width: 480px) { .movie-card { gap: 1rem; padding: .9rem 1.2rem; } }
.movie-card:hover { border-color: #3b82f6; transform: translateX(5px); box-shadow: 0 2px 20px rgba(59,130,246,.12); }
@keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.movie-card:nth-child(1) { animation-delay: .04s; }
.movie-card:nth-child(2) { animation-delay: .08s; }
.movie-card:nth-child(3) { animation-delay: .12s; }
.movie-card:nth-child(4) { animation-delay: .16s; }
.movie-card:nth-child(5) { animation-delay: .20s; }
.movie-card:nth-child(6) { animation-delay: .24s; }
.movie-card:nth-child(7) { animation-delay: .28s; }
.movie-card:nth-child(8) { animation-delay: .32s; }
.movie-card:nth-child(9) { animation-delay: .36s; }
.movie-card:nth-child(10) { animation-delay: .40s; }
.movie-rank { font-family: 'Satoshi', 'DM Sans', sans-serif; font-weight: 900; font-size: clamp(1.1rem, 4vw, 1.4rem); color: #1a2d50; min-width: 1.8rem; line-height: 1; }
.movie-name { font-size: clamp(.85rem, 3.5vw, .98rem); font-weight: 400; color: #c8d8f0; letter-spacing: .01em; word-break: break-word; }
.movie-dot { width: 6px; height: 6px; border-radius: 50%; background: #3b82f6; margin-left: auto; flex-shrink: 0; }
.error-box { background: rgba(59,130,246,.07); border: 1px solid rgba(59,130,246,.25); border-radius: 8px; padding: .85rem 1rem; color: #60a5fa; font-size: clamp(.8rem, 3vw, .88rem); }
</style>
""", unsafe_allow_html=True)

# ── CSS Part 4: Footer & social ───────────────────────────────────────────────
st.markdown("""
<style>
.footer { text-align: center; margin-top: 3.5rem; padding-top: 1.8rem; border-top: 1px solid #0d1f3c; }
.footer-brand { font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; color: #7a9cc0; margin-bottom: .3rem; font-weight: 500; }
.footer-tagline { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #4a6080; margin-bottom: 1.2rem; }
.footer-tagline { font-size: .8rem; letter-spacing: .06em; color: #6a8ab0; margin-bottom: 1.4rem; }
.footer-socials { display: flex; justify-content: center; gap: .75rem; margin-bottom: .85rem; flex-wrap: wrap; }
.social-link { display: flex; align-items: center; justify-content: center; width: 42px; height: 42px; border-radius: 50%; background: #080f1e; border: 1px solid #1a2d50; color: #4d8eff; text-decoration: none; transition: background .2s, border-color .2s, color .2s, transform .2s, box-shadow .2s; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
.social-link:hover { background: #1d4ed8; border-color: #3b82f6; color: #fff; transform: translateY(-3px); box-shadow: 0 4px 16px rgba(29,78,216,.4); }
.social-link:active { transform: translateY(0); background: #1e40af; }
.footer-name { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #4d6a8a; margin-top: .4rem; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    df        = pickle.load(open('df.pkl',           'rb'))
    indices   = pickle.load(open('indices.pkl',      'rb'))
    tfidf     = pickle.load(open('tfidf.pkl',        'rb'))
    tfidf_mat = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    return df, indices, tfidf, tfidf_mat

df, indices, tfidf, tfidf_matrix = load_data()

# ── Recommendation logic ─────────────────────────────────────────────────────
def recommend(movie_name, num=5):
    if movie_name not in indices:
        return None
    idx        = indices[movie_name]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_idx    = sim_scores.argsort()[-(num + 1):-1][::-1]
    return df['title'].iloc[sim_idx].tolist()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Content-based filtering</div>
    <div class="hero-title">Movie<br><span>Recommendation</span><br>System</div>
    <div class="hero-sub">Pick a movie. Discover films you'll love.</div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Controls ─────────────────────────────────────────────────────────────────
st.markdown('<div class="select-label">Select a movie</div>', unsafe_allow_html=True)
selected_movie = st.selectbox(
    label="Select a movie",
    options=df['title'].values,
    label_visibility="collapsed",
)

st.markdown('<div class="select-label" style="margin-top:1.2rem;">Number of recommendations</div>', unsafe_allow_html=True)
num_recs = st.slider(
    label="Number of recommendations",
    min_value=3,
    max_value=10,
    value=5,
    label_visibility="collapsed",
)

find_btn = st.button("FIND RECOMMENDATIONS →")

# ── Results ──────────────────────────────────────────────────────────────────
if find_btn:
    recs = recommend(selected_movie, num_recs)
    if recs is None:
        st.markdown('<div class="error-box">Movie not found in the dataset.</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-label">Movie Recommendations</div>
        <div class="results-header">Top Picks for You</div>
        """, unsafe_allow_html=True)

        cards_html = ""
        for i, movie in enumerate(recs, 1):
            cards_html += f'<div class="movie-card"><div class="movie-rank">{str(i).zfill(2)}</div><div class="movie-name">{movie}</div><div class="movie-dot"></div></div>'
        st.markdown(cards_html, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-tagline">Join my journey and let's explore together.</div>
    <div class="footer-socials">
        <a href="https://www.linkedin.com/in/vishal-singh-here/" target="_blank" class="social-link" title="LinkedIn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
        </a>
        <a href="https://github.com/VishalIndevp" target="_blank" class="social-link" title="GitHub">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
        </a>
        <a href="https://x.com/vishalindev" target="_blank" class="social-link" title="X">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.748l7.73-8.835L1.254 2.25H8.08l4.261 5.631 5.903-5.631zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
        </a>
        <a href="https://www.instagram.com/vishalindev" target="_blank" class="social-link" title="Instagram">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/></svg>
        </a>
    </div>
    <div class="footer-name">Made by Vishal Singh</div>
</div>
""", unsafe_allow_html=True)