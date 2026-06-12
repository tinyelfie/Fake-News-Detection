"""
Page 1 — Home
Project overview, dataset info, model info cards.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import load_eda_summary, load_all_metrics, PALETTE

st.set_page_config(page_title="Home — Fake News Detector", page_icon="", layout="wide")

# ── Inject shared CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #FFF8F3 !important; color: #4A3F35 !important; }
section[data-testid="stSidebar"] { background: #4A3F35 !important; }
section[data-testid="stSidebar"] * { color: #FFF8F3 !important; }
.stButton > button { background: #FF8FAB !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
[data-testid="stMetric"] { background: white !important; border-radius: 12px !important; padding: 1rem !important; border: 1px solid #FFD6E0 !important; box-shadow: 0 2px 10px rgba(107,144,128,0.12) !important; }
[data-testid="stMetricLabel"] { color: #FF8FAB !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #4A3F35 !important; font-weight: 800 !important; }
.card { background: white; border-radius: 14px; padding: 1.5rem 1.8rem; border: 1px solid #FFD6E0; box-shadow: 0 4px 15px rgba(107,144,128,0.10); margin-bottom: 1rem; transition: transform 0.2s, box-shadow 0.2s; }
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(107,144,128,0.20); }
.card h3 { color: #FF8FAB; margin-bottom: 0.4rem; }
.section-header { font-size: 1.6rem; font-weight: 800; color: #4A3F35; border-left: 5px solid #E63946; padding-left: 0.8rem; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: #4A3F35;
            border-radius: 20px; padding: 3rem 2.5rem; margin-bottom: 2rem; text-align:center;">
    <div style="font-size:3rem; margin-bottom:0.8rem;"></div>
    <h1 style="color:#FFF8F3; font-size:2.6rem; font-weight:800; margin:0 0 0.5rem 0;">
        Fake News Detection System
    </h1>
    <p style="color:#FFD6E0; font-size:1.1rem; max-width:650px; margin:0 auto;">
        An end-to-end ML/NLP pipeline classifying news articles as
        <strong style="color:#FFF8F3;">Fake</strong> or
        <strong style="color:#FFD6E0;">Real</strong> using six different
        AI architectures — from classical ML to transformers.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Dataset Stats ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Dataset Overview</div>', unsafe_allow_html=True)
eda = load_eda_summary()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Articles",  f"{eda.get('total_articles', '~44,000'):,}" if isinstance(eda.get('total_articles'), int) else "~44,000")
c2.metric("Fake Articles",   f"{eda.get('fake_articles',  '~23,481'):,}" if isinstance(eda.get('fake_articles'), int) else "~23,481")
c3.metric("Real Articles",   f"{eda.get('real_articles',  '~21,417'):,}" if isinstance(eda.get('real_articles'), int) else "~21,417")
c4.metric("Avg. Article Length", f"{eda.get('mean_length', '~400'):.0f} words" if isinstance(eda.get('mean_length'), (int, float)) else "~400 words")

st.markdown("<br>", unsafe_allow_html=True)

# ── Phase Cards ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Project Pipeline</div>', unsafe_allow_html=True)

phases = [
    ("Phase 1", "🧹 Data Preprocessing", "Load, clean, EDA — lemmatization, stopword removal, TF-IDF prep."),
    ("Phase 2", "🧮 Traditional ML", "TF-IDF + Logistic Regression + Multinomial Naive Bayes."),
    ("Phase 3", "🧠 Deep Learning", "BiLSTM and GRU architectures with early stopping."),
    ("Phase 4", "🤗 Transformers", "BERT & RoBERTa fine-tuned with HuggingFace + PyTorch."),
    ("Phase 5", " Comparative Analysis", "Side-by-side model comparison, confusion matrices, ROC curves."),
    ("Phase 6", "🌐 Streamlit App", "Interactive multi-page web app for prediction & analysis."),
]

cols = st.columns(3)
for i, (tag, title, desc) in enumerate(phases):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.75rem; font-weight:700; color:#E63946; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:0.3rem;">{tag}</div>
            <h3 style="font-size:1rem;">{title}</h3>
            <p style="font-size:0.85rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Model Info Cards ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Models at a Glance</div>', unsafe_allow_html=True)

metrics_df = load_all_metrics()

models_info = [
    ("Logistic Regression", "Classical ML", "#FF8FAB",
     "TF-IDF + linear classifier. Blazing fast, highly interpretable."),
    ("Naive Bayes", "Classical ML", "#FFD6E0",
     "Probabilistic bag-of-words model. Ideal for high-throughput screening."),
    ("BiLSTM", "Deep Learning", "#E63946",
     "Bidirectional LSTM captures sequential context in both directions."),
    ("GRU", "Deep Learning", "#FFF8F3",
     "Gated Recurrent Unit — faster than LSTM with competitive accuracy."),
    ("BERT", "Transformer", "#4A3F35",
     "bert-base-uncased fine-tuned. Deep contextual language understanding."),
    ("RoBERTa", "Transformer", "#FFF8F3",
     "Robustly optimized BERT pretraining. State-of-the-art NLP performance."),
]

cols2 = st.columns(3)
for i, (mname, mtype, color, desc) in enumerate(models_info):
    acc_str = "—"
    if not metrics_df.empty:
        row = metrics_df[metrics_df['model'] == mname]
        if not row.empty and 'accuracy' in row.columns:
            acc_str = f"{row.iloc[0]['accuracy']:.2%}"

    with cols2[i % 3]:
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="font-size:1rem; margin:0;">{mname}</h3>
                <span style="background:{color}22; color:{color}; font-size:0.7rem;
                             font-weight:700; padding:0.2rem 0.6rem; border-radius:20px;">
                    {mtype}
                </span>
            </div>
            <p style="font-size:0.85rem; margin:0.5rem 0 0.3rem 0;">{desc}</p>
            <div style="font-size:0.9rem; font-weight:700; color:{color};">
                Accuracy: {acc_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ Tech Stack</div>', unsafe_allow_html=True)

techs = ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn",
         "Plotly", "Scikit-learn", "TensorFlow/Keras",
         "HuggingFace Transformers", "PyTorch", "Streamlit"]

badges = " ".join([
    f'<span style="background:white; border:1px solid #FFD6E0; color:#4A3F35; '
    f'padding:0.3rem 0.8rem; border-radius:20px; font-size:0.85rem; '
    f'font-weight:500; margin:0.2rem; display:inline-block;">{t}</span>'
    for t in techs
])
st.markdown(f'<div style="line-height:2.2;">{badges}</div>', unsafe_allow_html=True)
