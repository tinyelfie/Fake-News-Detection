"""
Streamlit App — Main Entrypoint
Fake News Detection System
"""

import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:        #FFF8F3;
    --surface:   #FFFFFF;
    --primary:   #FF8FAB;
    --secondary: #FFD6E0;
    --text:      #4A3F35;
    --accent:    #E63946;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #4A3F35 !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #FFF8F3 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] p {
    color: #FFD6E0 !important;
}

/* Page navigation links */
section[data-testid="stSidebar"] a {
    color: #FFD6E0 !important;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] a:hover {
    color: #FFF8F3 !important;
}

/* Buttons */
.stButton > button {
    background: #FF8FAB !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.8rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(107,144,128,0.4) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    border: 1px solid #FFD6E0 !important;
    box-shadow: 0 2px 10px rgba(107,144,128,0.12) !important;
}
[data-testid="stMetricLabel"] { color: #FF8FAB !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #4A3F35 !important; font-weight: 800 !important; }

/* Cards */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    border: 1px solid #FFD6E0;
    box-shadow: 0 4px 15px rgba(107,144,128,0.10);
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(107,144,128,0.20);
}
.card h3 { color: #FF8FAB; margin-bottom: 0.4rem; }
.card p  { color: #4A3F35; margin: 0; font-size: 0.9rem; }

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4A3F35;
    border-left: 5px solid #E63946;
    padding-left: 0.8rem;
    margin-bottom: 1.2rem;
}

/* Badge */
.badge-fake {
    display: inline-block;
    background: #E63946;
    color: white;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}
.badge-real {
    display: inline-block;
    background: #FF8FAB;
    color: white;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

/* Tables */
.dataframe { border-radius: 8px !important; overflow: hidden !important; }
thead tr th { background-color: #FF8FAB !important; color: white !important; }
tbody tr:nth-child(even) { background-color: #f0f5f2 !important; }

/* Progress bar */
.stProgress > div > div { background-color: #FF8FAB !important; }

/* Selectbox */
.stSelectbox > div > div {
    border-color: #FFD6E0 !important;
    border-radius: 8px !important;
}

/* Text area */
.stTextArea textarea {
    border-color: #FFD6E0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #FF8FAB !important;
    box-shadow: 0 0 0 2px rgba(107,144,128,0.25) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:2.8rem; margin-bottom:0.3rem;"></div>
        <div style="font-size:1.3rem; font-weight:800; color:#FFF8F3; letter-spacing:0.02em;">
            FakeDetect
        </div>
        <div style="font-size:0.8rem; color:#FFD6E0; margin-top:0.2rem;">
            AI-Powered News Analysis
        </div>
        <hr style="border-color:#FFD6E060; margin:1.2rem 0 0.5rem 0;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Navigate**
    -  Home
    -  News Prediction
    -  Model Comparison
    -  Batch Prediction
    -  Analytics Dashboard
    """)

    st.markdown("""
    <hr style="border-color:#FFD6E060; margin: 1rem 0;">
    <div style="font-size:0.75rem; color:#FFD6E0; text-align:center;">
        Built with  using<br>
        Python · Scikit-learn<br>
        TensorFlow · HuggingFace<br>
        Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Redirect to Home ──────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 4rem 0 2rem 0;">
    <div style="font-size:3.5rem; margin-bottom:1rem;"></div>
    <h1 style="font-size:2.5rem; font-weight:800; color:#4A3F35;">
        Fake News Detection System
    </h1>
    <p style="font-size:1.1rem; color:#FF8FAB; max-width:600px; margin:0 auto 2rem auto;">
        An end-to-end ML/NLP portfolio project comparing 6 models across
        Classical ML, Deep Learning, and Transformer architectures.
    </p>
    <p style="color:#E63946; font-weight:600;">
         Use the sidebar to navigate between pages
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="card">
        <h3> 6 Models</h3>
        <p>Logistic Regression, Naive Bayes, BiLSTM, GRU, BERT, RoBERTa</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="card">
        <h3> 44,000 Articles</h3>
        <p>Fake and Real News Dataset from Kaggle — balanced binary classification</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="card">
        <h3> Full Pipeline</h3>
        <p>Preprocessing → Training → Evaluation → Deployment</p>
    </div>""", unsafe_allow_html=True)
