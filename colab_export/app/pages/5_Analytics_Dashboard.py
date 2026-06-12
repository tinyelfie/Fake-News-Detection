"""
Page 5 — Analytics Dashboard
Dataset statistics and model performance visualizations using Plotly.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import load_all_metrics, load_eda_summary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EDA_DIR  = os.path.join(BASE_DIR, 'outputs', 'eda')

st.set_page_config(page_title="Analytics Dashboard — Fake News Detector",
                   page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #FFF8F3 !important; color: #4A3F35 !important; }
section[data-testid="stSidebar"] { background: #4A3F35 !important; }
section[data-testid="stSidebar"] * { color: #FFF8F3 !important; }
.section-header { font-size: 1.6rem; font-weight: 800; color: #4A3F35; border-left: 5px solid #E63946; padding-left: 0.8rem; margin-bottom: 1.2rem; }
[data-testid="stMetric"] { background: white !important; border-radius: 12px !important; padding: 1rem !important; border: 1px solid #FFD6E0 !important; box-shadow: 0 2px 10px rgba(107,144,128,0.12) !important; }
[data-testid="stMetricLabel"] { color: #FF8FAB !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #4A3F35 !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

PALETTE_LIST = ['#FF8FAB', '#FFD6E0', '#E63946', '#FFF8F3', '#4A3F35', '#FFF8F3']

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"> Analytics Dashboard</div>', unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
eda    = load_eda_summary()
df_met = load_all_metrics()

# ── Section 1: Dataset Statistics ────────────────────────────────────────────
st.markdown('<div class="section-header" style="font-size:1.1rem;"> Dataset Statistics</div>', unsafe_allow_html=True)

if eda:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Articles",        f"{eda.get('total_articles', 0):,}")
    c2.metric("Fake Articles",          f"{eda.get('fake_articles',  0):,}")
    c3.metric("Real Articles",          f"{eda.get('real_articles',  0):,}")
    c4.metric("Mean Article Length",    f"{eda.get('mean_length', 0):.0f} words")
    st.markdown("<br>", unsafe_allow_html=True)

    # Class distribution donut
    col_donut, col_hist = st.columns(2)
    with col_donut:
        st.markdown("**Class Distribution**")
        fig_donut = go.Figure(go.Pie(
            labels=['Fake', 'Real'],
            values=[eda.get('fake_articles', 0), eda.get('real_articles', 0)],
            hole=0.55,
            marker_colors=['#E63946', '#FF8FAB'],
            textinfo='label+percent+value',
            textfont=dict(family='Inter', size=12),
        ))
        fig_donut.update_layout(
            paper_bgcolor='#FFF8F3',
            font=dict(family='Inter', color='#4A3F35'),
            showlegend=True,
            legend=dict(font=dict(size=12)),
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig_donut.add_annotation(
            text=f"<b>{eda.get('total_articles', 0):,}</b><br>Articles",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#4A3F35', family='Inter'),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_hist:
        # Load cleaned data for length distribution
        cleaned_path = os.path.join(DATA_DIR, 'cleaned_data.csv')
        if os.path.exists(cleaned_path):
            st.markdown("**Article Length Distribution**")
            df_data = pd.read_csv(cleaned_path)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_data[df_data['label'] == 0]['article_length'],
                name='Fake',
                marker_color='#E63946',
                opacity=0.75,
                nbinsx=50,
            ))
            fig_hist.add_trace(go.Histogram(
                x=df_data[df_data['label'] == 1]['article_length'],
                name='Real',
                marker_color='#FF8FAB',
                opacity=0.75,
                nbinsx=50,
            ))
            fig_hist.update_layout(
                barmode='overlay',
                paper_bgcolor='#FFF8F3', plot_bgcolor='#FFFFFF',
                font=dict(family='Inter', color='#4A3F35'),
                xaxis=dict(title='Word Count', gridcolor='#FFD6E0'),
                yaxis=dict(title='Frequency',  gridcolor='#FFD6E0'),
                legend=dict(font=dict(size=11)),
                height=320,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Run Phase 1 to generate article length data.")
else:
    st.info("Run Phase 1 to generate EDA summary.")

# ── Section 2: EDA Images ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header" style="font-size:1.1rem;"> EDA Visualizations</div>', unsafe_allow_html=True)

eda_images = {
    'Class Distribution':      'class_distribution.png',
    'Article Length Distribution': 'article_length_distribution.png',
    'Top Words — Fake News':   'top_words_fake.png',
    'Top Words — Real News':   'top_words_real.png',
}

img_cols = st.columns(2)
for idx, (title, fname) in enumerate(eda_images.items()):
    fpath = os.path.join(EDA_DIR, fname)
    if os.path.exists(fpath):
        with img_cols[idx % 2]:
            st.markdown(f"**{title}**")
            st.image(fpath, use_container_width=True)

# ── Section 3: Model Performance ─────────────────────────────────────────────
if not df_met.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.1rem;"> Model Performance Overview</div>', unsafe_allow_html=True)

    # Grouped bar — all metrics side by side
    metrics_to_show = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df_met.columns]
    fig_grouped = go.Figure()
    bar_colors   = ['#FF8FAB', '#FFD6E0', '#E63946', '#FFF8F3']

    for metric, color in zip(metrics_to_show, bar_colors):
        fig_grouped.add_trace(go.Bar(
            name=metric.capitalize(),
            x=df_met['model'],
            y=df_met[metric],
            marker_color=color,
            text=[f'{v:.3f}' for v in df_met[metric]],
            textposition='outside',
        ))

    fig_grouped.update_layout(
        barmode='group',
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFF8F3',
        font=dict(family='Inter', color='#4A3F35'),
        yaxis=dict(range=[0.75, 1.05], title='Score', gridcolor='#FFD6E0'),
        xaxis=dict(title='Model'),
        legend=dict(font=dict(size=11)),
        height=450,
        margin=dict(l=40, r=40, t=30, b=60),
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    # Heatmap
    st.markdown("**Performance Heatmap**")
    heatmap_data = df_met.set_index('model')[metrics_to_show]
    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale=[
            [0.0, '#FFF8F3'],
            [0.5, '#FFD6E0'],
            [1.0, '#FF8FAB'],
        ],
        aspect='auto',
        text_auto='.4f',
        labels=dict(color='Score'),
    )
    fig_heat.update_layout(
        paper_bgcolor='#FFF8F3',
        font=dict(family='Inter', color='#4A3F35', size=12),
        height=320,
        margin=dict(l=40, r=40, t=30, b=40),
        coloraxis_colorbar=dict(title='Score'),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
