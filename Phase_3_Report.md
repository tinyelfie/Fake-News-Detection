# Technical Report: Fake News Detection
**Scope:** Phase 3 (Deep Learning - BiLSTM & GRU)

---

## Phase 3: Deep Learning

Phase 3 transitions from term-frequency vectors to sequential Deep Learning, utilizing Keras embeddings and Recurrent Neural Networks to capture the order and context of words.

### 1. Training Setup
- **Dataset Split:** 5,130 Train / 570 Validation / 1,425 Test
- **Architecture:** Both models utilized a standard sequential pipeline: Embedding Layer -> RNN Layer (BiLSTM/GRU) -> Dropout -> Dense Output
- **Training Constraints:** Models were trained for a maximum of 20 epochs with an EarlyStopping patience of 3 epochs to prevent overfitting.

### 2. Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **BiLSTM** | **0.9979** | 0.9979 | 0.9979 | 0.9979 | 45.45s |
| **GRU** | **0.9986** | 0.9986 | 0.9986 | 0.9986 | **32.60s** |

### 3. Analysis of Training Dynamics & Output
1. **Performance Improvements:** The Deep Learning models successfully beat the extremely high bar set by Logistic Regression (99.30%). The GRU hit **99.86%** accuracy, meaning it misclassified less than 2 articles out of every 1000. 
2. **BiLSTM Convergence:** The BiLSTM converged extraordinarily fast. By the end of Epoch 1, it already achieved 90.4% training accuracy, and by Epoch 3 it hit its global minimum (val_loss: 0.0118). Training halted at epoch 6. The bidirectional context immediately helped the model understand the text.
3. **The GRU Plateau:** The GRU exhibited a fascinating training dynamic. For the first 6 epochs, it was stuck in a local minimum/plateau, hovering at exactly **51% accuracy** (which equates to random guessing on a balanced dataset). However, at Epoch 7, the optimizer broke through the plateau, and accuracy immediately shot up to 66%, reaching 99% by Epoch 8.
4. **Efficiency:** As expected, the GRU (Gated Recurrent Unit) was much faster to train (32.6s) than the BiLSTM (45.4s) due to having fewer internal gates, while still achieving a marginally higher final accuracy.


 Phase 3 (Deep Learning) Key Analysis Highlights:
Breaking the Logistic Regression Ceiling: Despite Logistic Regression setting an incredibly high bar (99.30%), your Deep Learning models still managed to beat it, reaching 99.86% accuracy with the GRU!
The GRU's Training Struggle: An interesting phenomenon occurred while training your GRU model. For the first 6 epochs, the GRU was completely stuck in a learning plateau—it hovered at exactly 51% accuracy (which is equal to random guessing on a balanced dataset). Then, at Epoch 7, the optimizer finally "broke through" the math, and accuracy instantly shot up to 66%, reaching 99% by Epoch 8.
Speed vs. Power: As expected, the GRU was significantly faster to train (32 seconds) than the BiLSTM (45 seconds) because it has fewer internal mathematical gates, while still managing to output slightly better accuracy.