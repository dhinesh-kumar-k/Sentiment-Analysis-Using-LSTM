# Sentiment-Analysis-Using-LSTM
Arttifai Tech's Internship Repository (Sentiment Analysis Using LSTM)

## 🎯 Project Objective

- Build a deep learning model using LSTM for binary sentiment classification.
- Train the model on preprocessed IMDB movie reviews.
- Evaluate performance using accuracy and confusion matrix.
- Predict sentiment of custom text inputs.

├── INTERNSHIP_(Sentiment_Analysis_Using_LSTM).ipynb # Main Jupyter notebook
├── imdb.npz # Tokenized dataset
├── x_train.csv, y_train.csv, x_test.csv, y_test.csv # Converted CSV data


---

## 📦 Dataset

- Dataset: IMDB Movie Reviews
- Source: Keras built-in dataset (`tensorflow.keras.datasets.imdb`)
- Total Samples: 50,000 (25,000 for training and 25,000 for testing)
- Each review is encoded as a sequence of integers (word indices).

---

## 🧠 Model Architecture

- **Embedding Layer** – converts word indices to dense vectors.
- **LSTM Layer** – captures sequential dependencies in text.
- **Dense Output Layer** – with `sigmoid` activation for binary classification.

```python
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

✅ Evaluation Metrics
Accuracy: ~85% on test data.

Confusion Matrix: Analyzed false positives/negatives.

Visualized using matplotlib and seaborn.
