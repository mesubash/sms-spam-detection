import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os  
import sys  
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Ensure the models directory exists to save the trained models
os.makedirs("models", exist_ok=True)
print("✅ Created 'models' directory")

# Ensure the 'data/' directory exists
if not os.path.exists("data"):
    print("❌ Error: The 'data/' folder does not exist.")
    print("Please run 'data_preprocessing.py' first.")
    sys.exit(1)  # Exit the script with an error code

# Ensure the 'plots/' directory exists
if not os.path.exists("plots"):
    os.makedirs("plots")
    print("✅ Created 'plots/' directory")

# Load preprocessed data
try:
    df = pd.read_csv("data/merged_spam_ham.csv")
    print("✅ Dataset import completed!")
except FileNotFoundError:
    print("❌ Error: 'data/merged_spam_ham.csv' not found.")
    print("Please ensure 'data_preprocessing.py' has been run successfully.")
    sys.exit(1)  # Exit the script with an error code

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

y_test_pred_lstm = (model.predict(X_test_pad) > 0.6).astype(int)
print("LSTM Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_lstm):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_lstm):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred_lstm):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred_lstm):.4f}")

# Save models and vectorizers
model.save("models/sms_spam_model.h5")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(tokenizer, "models/tokenizer.pkl")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_lstm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Spam', 'Not Spam'], yticklabels=['Spam', 'Not Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png")
plt.show()
plt.close()
print("✅ Confusion Matrix plotted and saved as plots/confusion_matrix.png")

# ROC Curve
y_test_pred_proba = model.predict(X_test_pad)
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("plots/roc_curve.png")
plt.show()
plt.close()
print("✅ ROC Curve plotted and saved as plots/roc_curve.png")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
average_precision = average_precision_score(y_test, y_test_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig("plots/precision_recall_curve.png")
plt.show()
plt.close()
print("✅ Precision-Recall Curve plotted and saved as plots/precision_recall_curve.png")

# Training and Validation Loss/Accuracy Curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig("plots/training_validation_curves.png")
plt.show()
plt.close()
print("✅ Training and validation curves plotted and saved as plots/training_validation_curves.png")