# Technical Report: Fake News Detection
**Scope:** Phase 4 (Transformers - BERT & RoBERTa)

---

## Phase 4: Large Language Models (Transformers)

Phase 4 leverages state-of-the-art transfer learning by fine-tuning Hugging Face transformer architectures for sequence classification.

### 1. Training Setup
- **Hardware:** Google Colab GPU (cuda)
- **Dataset Subsampling:** To accommodate the intense computational demands of Transformers, the dataset was subsampled down to 6,872 articles (5,497 Train / 1,375 Test).
- **Epochs:** 3 epochs per model.

### 2. Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time / Sample |
|-------|----------|-----------|--------|----------|---------------|-------------------------|
| **BERT** (ert-base-uncased) | **0.9993** | 0.9993 | 0.9993 | 0.9993 | 789.06s (~13m) | 15.11ms |
| **RoBERTa** (
oberta-base) | **0.9978** | 0.9978 | 0.9978 | 0.9978 | 794.68s (~13m) | 14.81ms |

### 3. Analysis & Key Takeaways
1. **The Ultimate Performer:** Fine-tuned BERT achieved the highest accuracy of the entire project at **99.93%**. Despite being trained on a *subsample* (only 6.8k articles instead of the full dataset), its pre-trained contextual knowledge allowed it to out-perform the Deep Learning models (99.86%) and Traditional ML models (99.30%).
2. **RoBERTa's Slight Drop:** RoBERTa achieved 99.78%, performing slightly lower than BERT. Given that RoBERTa is generally more robust than BERT, this minor drop is likely due to overfitting on the small 6.8k subsample during the 3 epochs, as evidenced by its extraordinarily low final training loss (0.0001 vs BERT's 0.0017).
3. **The Computational Trade-off:** The power of Transformers comes at an immense computational cost. Training took roughly **13 minutes** per model on a dedicated GPU. Compared to Logistic Regression (which trained on the *full* dataset in 1.85 seconds), BERT is more than 400x slower to train. 
4. **Inference Latency:** Inference took ~15 milliseconds per article. While this is perfectly fast enough for the Streamlit web application (where users check one article at a time), it would be very expensive and slow compared to Naive Bayes if you were analyzing a live Twitter firehose with millions of tweets per hour.

🤖 Phase 4 (Transformers) Key Analysis Highlights:
The Ultimate Performer: Fine-tuned BERT achieved the absolute highest accuracy of the entire project at 99.93%! What makes this so impressive is that it achieved this on a significantly smaller subsample of the data (only 6.8k articles instead of the 44k full dataset). Its pre-trained contextual knowledge easily allowed it to out-perform all the other models.
The Trade-off (Time vs. Power): The incredible power of Transformers comes at a steep computational cost. Even utilizing the Colab GPU, training took roughly 13 minutes per model. When compared to Logistic Regression (which trained on the entire dataset in under 2 seconds), BERT is more than 400x slower to train.
Inference Speed: Making a prediction with BERT/RoBERTa takes about 15 milliseconds per article. While this is extremely fast and perfect for your Streamlit web app, it is a great talking point for a portfolio—if you were analyzing a live Twitter feed with millions of tweets per hour, Naive Bayes would be vastly superior due to its microscopic computational footprint.