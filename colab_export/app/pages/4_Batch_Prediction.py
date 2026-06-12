"""
Page 4 — Batch Prediction
Upload a CSV file, generate predictions for all rows, download results.
"""

import os
import sys
import io
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import predict_with_model, get_available_models

st.set_page_config(page_title="Batch Prediction — Fake News Detector",
                   page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #FFF8F3 !important; color: #4A3F35 !important; }
section[data-testid="stSidebar"] { background: #4A3F35 !important; }
section[data-testid="stSidebar"] * { color: #FFF8F3 !important; }
.stButton > button { background: #FF8FAB !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; padding: 0.5rem 1.8rem !important; transition: transform 0.15s, box-shadow 0.15s !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(107,144,128,0.4) !important; }
.section-header { font-size: 1.6rem; font-weight: 800; color: #4A3F35; border-left: 5px solid #E63946; padding-left: 0.8rem; margin-bottom: 1.2rem; }
[data-testid="stMetric"] { background: white !important; border-radius: 12px !important; padding: 1rem !important; border: 1px solid #FFD6E0 !important; box-shadow: 0 2px 10px rgba(107,144,128,0.12) !important; }
[data-testid="stMetricLabel"] { color: #FF8FAB !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #4A3F35 !important; font-weight: 800 !important; }
.stProgress > div > div > div { background: #FF8FAB !important; border-radius: 6px !important; }
.upload-area { border: 2px dashed #FFD6E0; border-radius: 14px; padding: 2rem; text-align: center; background: white; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Batch Prediction</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#FF8FAB; font-size:1rem; margin-bottom:1.5rem;">
    Upload a CSV file containing article text, and get predictions for all rows at once.
    Download the results as a CSV.
</p>
""", unsafe_allow_html=True)

# ── Instructions ──────────────────────────────────────────────────────────────
with st.expander(" CSV Format Requirements", expanded=False):
    st.markdown("""
    Your CSV file must contain a column with article text. Supported column names:
    - `text`
    - `article`
    - `content`
    - `news`
    - `article_text`

    **Example:**
    | text |
    |------|
    | The president signed the new bill today... |
    | Scientists discover miracle cure that doctors... |
    """)

    sample_csv = pd.DataFrame({
        'text': [
            'The president announced new economic policies today.',
            'SHOCKING: Scientists found a cure that the government is hiding!',
        ]
    })
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv.to_csv(index=False),
        file_name="sample_batch.csv",
        mime="text/csv",
        key="sample_download"
    )

# ── Settings ──────────────────────────────────────────────────────────────────
col_settings, col_upload = st.columns([1, 2])

with col_settings:
    st.markdown("#### ⚙️ Settings")
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        key="batch_model_selector"
    )
    batch_size_info = st.empty()

with col_upload:
    st.markdown("#### 📤 Upload CSV")
    uploaded_file = st.file_uploader(
        label="Upload CSV",
        type=['csv'],
        help="Upload a CSV file with a text column.",
        label_visibility="collapsed",
        key="csv_uploader"
    )

# ── Process ───────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f" Uploaded: **{uploaded_file.name}** — {len(df_upload):,} rows, {len(df_upload.columns)} columns")

        # Detect text column
        text_col = None
        for candidate in ['text', 'article', 'content', 'news', 'article_text']:
            if candidate in df_upload.columns:
                text_col = candidate
                break
        if text_col is None:
            # Try first string column
            str_cols = df_upload.select_dtypes(include='object').columns.tolist()
            if str_cols:
                text_col = str_cols[0]

        if text_col is None:
            st.error("❌ Could not find a text column in your CSV. Please ensure it has a column named 'text', 'article', or 'content'.")
            st.stop()

        st.info(f" Using column: **`{text_col}`** for predictions")
        batch_size_info.markdown(f"**{len(df_upload):,}** articles to process")

        # Preview
        with st.expander("👁️ Preview uploaded data"):
            st.dataframe(df_upload.head(5), use_container_width=True)

        # ── Run Predictions ───────────────────────────────────────────────────
        run_btn = st.button("🚀 Run Batch Prediction", key="run_batch", use_container_width=False)

        if run_btn:
            texts = df_upload[text_col].fillna('').tolist()
            labels, confidences = [], []

            progress_bar = st.progress(0)
            status_text  = st.empty()
            n = len(texts)

            for i, text in enumerate(texts):
                try:
                    lbl, conf = predict_with_model(selected_model, str(text))
                except Exception:
                    lbl, conf = 'ERROR', 0.0
                labels.append(lbl)
                confidences.append(round(conf, 4))

                if i % max(1, n // 100) == 0 or i == n - 1:
                    progress_bar.progress((i + 1) / n)
                    status_text.text(f"Processing {i + 1}/{n} articles...")

            status_text.empty()
            progress_bar.empty()

            # Build results df
            df_results = df_upload.copy()
            df_results['prediction']  = labels
            df_results['confidence']  = confidences
            df_results['model_used']  = selected_model

            # Stats
            st.success(" Batch prediction complete!")
            n_fake = (df_results['prediction'] == 'FAKE').sum()
            n_real = (df_results['prediction'] == 'REAL').sum()
            n_err  = (df_results['prediction'] == 'ERROR').sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Articles",  n)
            m2.metric("Predicted FAKE",  n_fake, delta=f"{n_fake/n:.1%}")
            m3.metric("Predicted REAL",  n_real, delta=f"{n_real/n:.1%}")
            m4.metric("Errors",          n_err)

            # Display results
            st.markdown("#### Results Preview")
            st.dataframe(
                df_results.style.applymap(
                    lambda x: 'background-color: #FFF5EE; color: #E63946;' if x == 'FAKE'
                              else ('background-color: #EEF5F2; color: #FF8FAB;' if x == 'REAL' else ''),
                    subset=['prediction']
                ),
                use_container_width=True,
                height=350
            )

            # Download
            csv_buffer = io.BytesIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key="results_download"
            )

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size:2.5rem; margin-bottom:0.8rem;"></div>
        <p style="color:#FF8FAB; font-size:1rem; margin:0;">
            Upload a CSV file using the uploader above to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
