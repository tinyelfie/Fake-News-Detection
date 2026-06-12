"""
Shared utilities for the Streamlit application.
"""

import os
import re
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Color Palette ──────────────────────────────────────────────────────────────
PALETTE = {
    'background': '#FFF8F3',
    'surface':    '#FFFFFF',
    'primary':    '#FF8FAB',
    'secondary':  '#FFD6E0',
    'text':       '#4A3F35',
    'accent':     '#E63946',
}

MODEL_DISPLAY_NAMES = [
    'Logistic Regression',
    'Naive Bayes',
    'BiLSTM',
    'GRU',
    'BERT',
    'RoBERTa',
]

# ── Text Cleaning (mirrors Phase 1) ───────────────────────────────────────────
def clean_text_for_inference(text: str) -> str:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem   import WordNetLemmatizer

    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('omw-1.4',   quiet=True)

    stop_words = set(stopwords.words('english'))
    lem        = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>',                 '', text)
    text = re.sub(r'[^a-z\s]',             '', text)
    text = re.sub(r'\s+',                  ' ', text).strip()
    tokens = [lem.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


# ── Load Models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tfidf():
    path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_lr():
    path = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_nb():
    path = os.path.join(MODEL_DIR, 'naive_bayes.joblib')
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_keras_tokenizer():
    path = os.path.join(MODEL_DIR, 'keras_tokenizer.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_bilstm():
    import tensorflow as tf
    path = os.path.join(MODEL_DIR, 'bilstm_model.keras')
    return tf.keras.models.load_model(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_gru():
    import tensorflow as tf
    path = os.path.join(MODEL_DIR, 'gru_model.keras')
    return tf.keras.models.load_model(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_bert():
    bert_dir = os.path.join(MODEL_DIR, 'bert_finetuned')
    if not os.path.exists(bert_dir):
        return None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(bert_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    return tok, mdl


@st.cache_resource(show_spinner=False)
def load_roberta():
    rdir = os.path.join(MODEL_DIR, 'roberta_finetuned')
    if not os.path.exists(rdir):
        return None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(rdir)
    mdl = AutoModelForSequenceClassification.from_pretrained(rdir)
    return tok, mdl


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_with_model(model_name: str, raw_text: str):
    """
    Returns (label: str, confidence: float) or raises on missing model.
    label ∈ {'FAKE', 'REAL'}
    confidence ∈ [0.0, 1.0]
    """
    cleaned = clean_text_for_inference(raw_text)

    if model_name == 'Logistic Regression':
        tfidf = load_tfidf()
        lr    = load_lr()
        if tfidf is None or lr is None:
            raise FileNotFoundError("Logistic Regression model not found. Run Phase 2 first.")
        vec   = tfidf.transform([cleaned])
        proba = lr.predict_proba(vec)[0]
        label = 'REAL' if np.argmax(proba) == 1 else 'FAKE'
        conf  = float(np.max(proba))

    elif model_name == 'Naive Bayes':
        tfidf = load_tfidf()
        nb    = load_nb()
        if tfidf is None or nb is None:
            raise FileNotFoundError("Naive Bayes model not found. Run Phase 2 first.")
        vec   = tfidf.transform([cleaned])
        proba = nb.predict_proba(vec)[0]
        label = 'REAL' if np.argmax(proba) == 1 else 'FAKE'
        conf  = float(np.max(proba))

    elif model_name == 'BiLSTM':
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        tok   = load_keras_tokenizer()
        bilstm = load_bilstm()
        if tok is None or bilstm is None:
            raise FileNotFoundError("BiLSTM model not found. Run Phase 3 first.")
        seq   = pad_sequences(tok.texts_to_sequences([cleaned]),
                              maxlen=500, padding='post', truncating='post')
        proba = float(bilstm.predict(seq, verbose=0)[0][0])
        label = 'REAL' if proba >= 0.5 else 'FAKE'
        conf  = proba if proba >= 0.5 else 1 - proba

    elif model_name == 'GRU':
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        tok = load_keras_tokenizer()
        gru = load_gru()
        if tok is None or gru is None:
            raise FileNotFoundError("GRU model not found. Run Phase 3 first.")
        seq   = pad_sequences(tok.texts_to_sequences([cleaned]),
                              maxlen=500, padding='post', truncating='post')
        proba = float(gru.predict(seq, verbose=0)[0][0])
        label = 'REAL' if proba >= 0.5 else 'FAKE'
        conf  = proba if proba >= 0.5 else 1 - proba

    elif model_name == 'BERT':
        import torch
        tok, mdl = load_bert()
        if tok is None or mdl is None:
            raise FileNotFoundError("BERT model not found. Run Phase 4 first.")
        inputs = tok(raw_text, return_tensors='pt', max_length=256,
                     truncation=True, padding=True)
        with torch.no_grad():
            logits = mdl(**inputs).logits
        proba = torch.softmax(logits, dim=1)[0].numpy()
        label = 'REAL' if np.argmax(proba) == 1 else 'FAKE'
        conf  = float(np.max(proba))

    elif model_name == 'RoBERTa':
        import torch
        tok, mdl = load_roberta()
        if tok is None or mdl is None:
            raise FileNotFoundError("RoBERTa model not found. Run Phase 4 first.")
        inputs = tok(raw_text, return_tensors='pt', max_length=256,
                     truncation=True, padding=True)
        with torch.no_grad():
            logits = mdl(**inputs).logits
        proba = torch.softmax(logits, dim=1)[0].numpy()
        label = 'REAL' if np.argmax(proba) == 1 else 'FAKE'
        conf  = float(np.max(proba))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return label, conf


# ── Load Metrics ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_metrics():
    all_metrics = []
    for fname in ['ml_metrics.json', 'dl_metrics.json', 'transformer_metrics.json']:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                all_metrics.extend(json.load(f))
    return pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_eda_summary():
    path = os.path.join(DATA_DIR, 'eda_summary.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── Model Availability ────────────────────────────────────────────────────────
def get_available_models():
    available = []
    if os.path.exists(os.path.join(MODEL_DIR, 'logistic_regression.joblib')):
        available.append('Logistic Regression')
    if os.path.exists(os.path.join(MODEL_DIR, 'naive_bayes.joblib')):
        available.append('Naive Bayes')
    if os.path.exists(os.path.join(MODEL_DIR, 'bilstm_model.keras')):
        available.append('BiLSTM')
    if os.path.exists(os.path.join(MODEL_DIR, 'gru_model.keras')):
        available.append('GRU')
    if os.path.exists(os.path.join(MODEL_DIR, 'bert_finetuned')):
        available.append('BERT')
    if os.path.exists(os.path.join(MODEL_DIR, 'roberta_finetuned')):
        available.append('RoBERTa')
    return available if available else MODEL_DISPLAY_NAMES
