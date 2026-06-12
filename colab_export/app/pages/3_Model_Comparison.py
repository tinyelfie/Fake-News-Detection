"""
Page 3 — Model Comparison
Performance metrics, charts, model rankings, ROC curves.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def hex_to_rgba(h, a):
    h = h.lstrip('#')
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import load_all_metrics

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMP_DIR   = os.path.join(BASE_DIR, 'outputs', 'comparison')
EDA_DIR    = os.path.join(BASE_DIR, 'outputs', 'eda')

st.set_page_config(page_title="Model Comparison — Fake News Detector",
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
st.markdown('<div class="section-header"> Model Comparison</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#FF8FAB; font-size:1rem; margin-bottom:1.5rem;">
    Side-by-side comparison of all 6 models across accuracy, precision, recall, F1, and training time.
</p>
""", unsafe_allow_html=True)

# ── Load Metrics ──────────────────────────────────────────────────────────────
df = load_all_metrics()

if df.empty:
    st.warning(" No metrics found. Please run the training phases first (Phase 2–4).")
    st.stop()

# ── Top Metrics ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="font-size:1.1rem;"> Best Model per Metric</div>', unsafe_allow_html=True)
cols_top = st.columns(4)
for col, metric in zip(cols_top, ['accuracy', 'precision', 'recall', 'f1']):
    if metric in df.columns:
        best_row  = df.loc[df[metric].idxmax()]
        col.metric(
            label=metric.capitalize(),
            value=f"{best_row[metric]:.4f}",
            delta=f"{best_row['model']}",
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics Bar Chart (Plotly) ────────────────────────────────────────────────
st.markdown('<div class="section-header" style="font-size:1.1rem;"> Metrics Comparison</div>', unsafe_allow_html=True)

metric_options = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df.columns]
selected_metric = st.selectbox("Select metric to display", metric_options,
                                format_func=str.capitalize, key="metric_select")

fig_bar = go.Figure()
for i, row in df.iterrows():
    fig_bar.add_trace(go.Bar(
        name=row['model'],
        x=[row['model']],
        y=[row.get(selected_metric, 0)],
        marker_color=PALETTE_LIST[i % len(PALETTE_LIST)],
        text=[f"{row.get(selected_metric, 0):.4f}"],
        textposition='outside',
        textfont=dict(size=12, color='#4A3F35'),
    ))

fig_bar.update_layout(
    showlegend=False,
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='#FFF8F3',
    font=dict(family='Inter', color='#4A3F35'),
    yaxis=dict(title=selected_metric.capitalize(), autorange=True,
               gridcolor='#FFD6E0', gridwidth=0.5),
    xaxis=dict(title='Model'),
    margin=dict(l=40, r=40, t=30, b=40),
    height=400,
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="font-size:1.1rem;"> Radar Chart — Multi-Metric</div>', unsafe_allow_html=True)
metrics_radar = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df.columns]

fig_radar = go.Figure()
for i, row in df.iterrows():
    vals = [row.get(m, 0) for m in metrics_radar]
    vals.append(vals[0])   # close polygon
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=metrics_radar + [metrics_radar[0]],
        fill='toself',
        name=row['model'],
        line_color=PALETTE_LIST[i % len(PALETTE_LIST)],
        fillcolor=hex_to_rgba(PALETTE_LIST[i % len(PALETTE_LIST)], 0.2),
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0.8, 1.0],
                        gridcolor='#FFD6E0', linecolor='#FFD6E0'),
        bgcolor='#FFFFFF',
    ),
    showlegend=True,
    legend=dict(font=dict(family='Inter', size=11)),
    paper_bgcolor='#FFF8F3',
    font=dict(family='Inter', color='#4A3F35'),
    height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Training Time ─────────────────────────────────────────────────────────────
if 'training_time_sec' in df.columns:
    st.markdown('<div class="section-header" style="font-size:1.1rem;"> Training Time</div>', unsafe_allow_html=True)
    fig_time = px.bar(
        df, x='model', y='training_time_sec',
        color='model',
        color_discrete_sequence=PALETTE_LIST,
        labels={'training_time_sec': 'Training Time (seconds)', 'model': 'Model'},
        text='training_time_sec',
    )
    fig_time.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
    fig_time.update_layout(
        showlegend=False,
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFF8F3',
        font=dict(family='Inter', color='#4A3F35'),
        yaxis=dict(title='Seconds', gridcolor='#FFD6E0'),
        margin=dict(l=40, r=40, t=30, b=40),
        height=350,
    )
    st.plotly_chart(fig_time, use_container_width=True)

# ── Full Metrics Table ────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="font-size:1.1rem;"> Full Metrics Table</div>', unsafe_allow_html=True)

display_cols = [c for c in ['model', 'accuracy', 'precision', 'recall', 'f1',
                              'training_time_sec', 'inference_time_ms_sample']
                if c in df.columns]
display_df = df[display_cols].copy()
display_df = display_df.sort_values('f1', ascending=False).reset_index(drop=True)
display_df.index = range(1, len(display_df) + 1)
display_df.index.name = 'Rank'

st.dataframe(
    display_df.style.format({
        'accuracy':  '{:.4f}',
        'precision': '{:.4f}',
        'recall':    '{:.4f}',
        'f1':        '{:.4f}',
        'training_time_sec':         '{:.1f}s',
        'inference_time_ms_sample':  '{:.4f}ms',
    }).background_gradient(
        subset=[c for c in ['accuracy', 'f1'] if c in display_df.columns],
        cmap='Greens',
    ),
    use_container_width=True,
)

# ── Saved Comparison Images ───────────────────────────────────────────────────
if os.path.exists(COMP_DIR):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.1rem;"> Comparison Plots</div>', unsafe_allow_html=True)

    images = {
        'Combined Metrics':         'combined_metrics.png',
        'Confusion Matrices':       'confusion_matrices_all.png',
        'ROC Curves':               'roc_curves_all.png',
        'Model Ranking':            'model_ranking.png',
    }
    img_cols = st.columns(2)
    for idx, (title, fname) in enumerate(images.items()):
        fpath = os.path.join(COMP_DIR, fname)
        if os.path.exists(fpath):
            with img_cols[idx % 2]:
                st.markdown(f"**{title}**")
                st.image(fpath, use_container_width=True)
