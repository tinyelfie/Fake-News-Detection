# 📰 Fake News Detection — End-to-End ML & NLP

## Overview
This project is a comprehensive end-to-end Machine Learning and Natural Language Processing pipeline designed to classify articles as "Fake" or "Real" news. It trains and compares multiple models across three different paradigms: Classical Machine Learning, Deep Learning (RNNs), and Large Language Models (Transformers). The project concludes with a fully functional Streamlit web application.

## 🚀 Setup & Execution

### Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Running the App
The interactive web application can be launched using Streamlit:
```bash
streamlit run app/app.py
```
*(Note: If running in Google Colab, you can use `ngrok` or `localtunnel` to expose the port).*

---

## 📊 Comprehensive Analysis & Findings

The project is broken down into five distinct phases of analysis. Below are the core findings from the dataset and the trained models.

### 1. Data Preprocessing & EDA (Phase 1)
- **Extremely Balanced Dataset:** The data consists of 7,125 highly clean articles (after deduplication), perfectly balanced between Fake and Real news.
- **Aggressive Compression:** The custom NLP cleaning pipeline (lowercasing, punctuation removal, stopword removal, and lemmatization) compressed the mean article length from **422 words down to 238 tokens**, stripping massive amounts of noise and drastically speeding up training.

### 2. Traditional Machine Learning (Phase 2)
- **Linearly Separable:** Logistic Regression achieved an incredibly high **99.30% accuracy**. This proves that the core vocabulary of fake news writers is highly distinct from real news journalists in this dataset.
- **Speed:** Naive Bayes trained in **0.01 seconds** while Logistic Regression took **1.85 seconds**. These classical models are blazing fast and set a very high bar for the Neural Networks to beat.

### 3. Deep Learning (Phase 3)
- **The RNN "Sweet Spot":** The BiLSTM (99.79%) and GRU (99.86%) successfully beat the Logistic Regression baseline by capturing the sequential context of words. 
- **Learning Dynamics:** The GRU exhibited a fascinating training dynamic where it was stuck at 51% accuracy (random guessing) for 6 epochs before instantly shooting up to 99% accuracy in the 7th epoch as the optimizer broke through the plateau.

### 4. Transformers / LLMs (Phase 4)
- **The Ultimate Performer:** Fine-tuned BERT achieved the highest accuracy of the entire project at **99.93%**, despite being trained on a much smaller 6.8k article subsample. RoBERTa achieved 99.78%.
- **The Trade-off:** While extremely powerful, Transformers took roughly **13 minutes** to train on a dedicated GPU (over 400x slower than Logistic Regression).

### 5. Final Leaderboard & Deployment Strategy (Phase 5)

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **BERT** | **99.93%** | 0.9993 | ~13.1 min |
| **GRU** | 99.86% | 0.9986 | 32.60 sec |
| **BiLSTM** | 99.79% | 0.9979 | 45.45 sec |
| **RoBERTa** | 99.78% | 0.9978 | ~13.2 min |
| **Logistic Regression**| 99.30% | 0.9930 | 1.85 sec |
| **Naive Bayes** | 97.47% | 0.9747 | 0.01 sec |

#### 💡 Hybrid Production Architecture
In a real-world scenario (e.g., monitoring a live Twitter firehose), a two-tiered system is the most efficient:
1. **Tier 1 (The Filter):** Use **Logistic Regression** to parse 99% of the bulk traffic due to its near-zero latency and minimal computational footprint.
2. **Tier 2 (The Arbiter):** Use **BERT** exclusively for difficult, highly-viral, or flagged borderline articles to leverage its deep semantic understanding while saving massive amounts of GPU costs.

---

## 🛠 Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Scikit-learn · TensorFlow/Keras · Hugging Face Transformers · PyTorch · Streamlit

## Dataset Link 
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
