"""
Page 2 — News Prediction
User pastes article text, selects a model, gets Fake/Real prediction + confidence.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import predict_with_model, get_available_models

st.set_page_config(page_title="News Prediction — Fake News Detector",
                   page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #FFF8F3 !important; color: #4A3F35 !important; }
section[data-testid="stSidebar"] { background: #4A3F35 !important; }
section[data-testid="stSidebar"] * { color: #FFF8F3 !important; }
.stButton > button { background: #FF8FAB !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; padding: 0.55rem 2rem !important; transition: transform 0.15s, box-shadow 0.15s !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(107,144,128,0.4) !important; }
.stSelectbox > div > div { border-color: #FFD6E0 !important; border-radius: 8px !important; }
.stTextArea textarea { border-color: #FFD6E0 !important; border-radius: 10px !important; }
.stTextArea textarea:focus { border-color: #FF8FAB !important; box-shadow: 0 0 0 2px rgba(107,144,128,0.25) !important; }
.section-header { font-size: 1.6rem; font-weight: 800; color: #4A3F35; border-left: 5px solid #E63946; padding-left: 0.8rem; margin-bottom: 1.2rem; }
.result-box { border-radius: 16px; padding: 2rem 2.5rem; text-align: center; margin-top: 1.5rem; }
.badge-fake { display: inline-block; background: #E63946; color: white; padding: 0.5rem 1.8rem; border-radius: 30px; font-size: 1.8rem; font-weight: 800; letter-spacing: 0.08em; }
.badge-real { display: inline-block; background: #FF8FAB; color: white; padding: 0.5rem 1.8rem; border-radius: 30px; font-size: 1.8rem; font-weight: 800; letter-spacing: 0.08em; }
.stProgress > div > div > div { background: #FF8FAB !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> News Prediction</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#FF8FAB; font-size:1rem; margin-bottom:1.5rem;">
    Paste any news article below, choose a model, and get an instant prediction
    with a confidence score.
</p>
""", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
available_models = get_available_models()

col_input, col_settings = st.columns([2, 1])

with col_settings:
    st.markdown("#### ⚙️ Settings")
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        help="Choose which AI model to use for prediction.",
        key="model_selector"
    )
    st.markdown(f"""
    <div style="background:white; border:1px solid #FFD6E0; border-radius:10px;
                padding:1rem; margin-top:0.5rem; font-size:0.85rem; color:#4A3F35;">
        <strong style="color:#FF8FAB;">About {selected_model}</strong><br>
        { {
            'Logistic Regression': 'Fast linear classifier on TF-IDF features. Best for quick, interpretable results.',
            'Naive Bayes':         'Probabilistic model on word frequencies. Extremely fast inference.',
            'BiLSTM':              'Bidirectional LSTM captures sequential patterns in both directions.',
            'GRU':                 'Gated Recurrent Unit — compact yet powerful sequence model.',
            'BERT':                'bert-base-uncased fine-tuned. Deep contextual understanding.',
            'RoBERTa':             'roberta-base fine-tuned. State-of-the-art performance.',
        }.get(selected_model, '') }
    </div>
    """, unsafe_allow_html=True)

with col_input:
    st.markdown("#### 📄 Article Text")
    article_text = st.text_area(
        label="Paste your news article here",
        height=280,
        placeholder="Paste the full article text here — the more text, the better the prediction...",
        label_visibility="collapsed",
        key="article_input"
    )
    word_count = len(article_text.split()) if article_text.strip() else 0
    st.caption(f"📝 Word count: **{word_count}** words")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
center_col = st.columns([1, 2, 1])[1]
with center_col:
    predict_btn = st.button(" Analyze Article", use_container_width=True, key="predict_btn")

if predict_btn:
    if not article_text.strip():
        st.warning(" Please paste some article text before predicting.", icon="")
    elif word_count < 10:
        st.warning(" Article seems too short. Please provide more text for accurate predictions.", icon="")
    else:
        with st.spinner(f"Analyzing with {selected_model}..."):
            try:
                label, confidence = predict_with_model(selected_model, article_text)

                is_fake    = label == 'FAKE'
                bg_color   = '#FFF5EE' if is_fake else '#EEF5F2'
                border_col = '#E63946' if is_fake else '#FF8FAB'
                badge_cls  = 'badge-fake' if is_fake else 'badge-real'
                icon       = '🚨' if is_fake else ''
                verdict    = 'This article appears to be FAKE' if is_fake else 'This article appears to be REAL'

                st.markdown(f"""
                <div class="result-box" style="background:{bg_color}; border: 2px solid {border_col};">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">{icon}</div>
                    <span class="{badge_cls}">{label}</span>
                    <p style="color:#4A3F35; font-size:1rem; margin-top:1rem; font-weight:500;">
                        {verdict}
                    </p>
                    <p style="color:#FF8FAB; font-size:0.9rem;">
                        Model: <strong>{selected_model}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Score: {confidence:.1%}**")
                st.progress(confidence)

                # Additional info
                st.markdown(f"""
                <div style="background:white; border:1px solid #FFD6E0; border-radius:10px;
                            padding:1rem 1.5rem; margin-top:1rem; font-size:0.85rem;">
                    <strong style="color:#FF8FAB;">Analysis Details</strong><br>
                    <span style="color:#4A3F35;">
                    📌 Model: {selected_model}<br>
                    📏 Article length: {word_count} words<br>
                    🎯 Confidence: {confidence:.4f} ({confidence:.1%})<br>
                    🏷️ Verdict: <strong>{'FAKE NEWS' if is_fake else 'REAL NEWS'}</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(f"🚫 Model not available: {e}", icon="🚫")
                st.info("Run the training phases first:\n```\npython src/phase2_traditional_ml.py\npython src/phase3_deep_learning.py\npython src/phase4_transformers.py\n```")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}", icon="❌")

# ── Sample Articles ───────────────────────────────────────────────────────────
with st.expander(" Try a sample article"):
    st.markdown("**Sample Fake News (click to copy):**")
    fake_sample = ("BREAKING: Scientists discover that drinking coffee backwards "
                   "can cure all known diseases. The government has been hiding this "
                   "secret for decades. Thousands of doctors confirm this revolutionary "
                   "finding that big pharma doesn't want you to know about.")
    st.code(fake_sample, language=None)

    st.markdown("**Sample Real News:**")
    real_sample = ("The Federal Reserve raised its benchmark interest rate by 0.25 "
                   "percentage points on Wednesday, as expected, bringing it to a "
                   "target range of 5.25% to 5.50%. Fed Chair Jerome Powell said the "
                   "central bank remains focused on bringing inflation back to its 2% target.")
    st.code(real_sample, language=None)
