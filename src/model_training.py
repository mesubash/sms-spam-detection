import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib

# Load preprocessed data
df = pd.read_csv("data/merged_spam_ham.csv")

# Split the dataset into train (80%), validation (10%), and test (10%)
print(f"Total dataset size: {len(df)}")
print("Splitting the dataset into train, validation, and test sets...")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(train_df['cleaned_message'])
X_val_vec = vectorizer.transform(val_df['cleaned_message'])
X_test_vec = vectorizer.transform(test_df['cleaned_message'])
y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Evaluate Naive Bayes Model
y_val_pred = nb_model.predict(X_val_vec)
y_test_pred = nb_model.predict(X_test_vec)
print("Naive Bayes Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")

# Bi-LSTM Model
MAX_WORDS = 5000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_df['cleaned_message'])
X_train_seq = tokenizer.texts_to_sequences(train_df['cleaned_message'])
X_val_seq = tokenizer.texts_to_sequences(val_df['cleaned_message'])
X_test_seq = tokenizer.texts_to_sequences(test_df['cleaned_message'])
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        score = tf.nn.softmax(inputs, axis=1)
        return inputs * score

model = tf.keras.Sequential([
    layers.Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(128, return_sequences=True)),
    AttentionLayer(),
    Dropout(0.6),
    Bidirectional(LSTM(64)),
    Dropout(0.6),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
class_weights = {0: 1, 1: 1.5}
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_val_pad, y_val),
    class_weight=class_weights,
    callbacks=[early_stopping]
)

y_test_pred_lstm = (model.predict(X_test_pad) > 0.3).astype(int)
print("LSTM Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_lstm):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_lstm):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred_lstm):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred_lstm):.4f}")

# Save models and vectorizers
model.save("models/sms_spam_model.h5")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(tokenizer, "models/tokenizer.pkl")