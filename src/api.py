from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import re

# Define the custom AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        score = tf.nn.softmax(inputs, axis=1)
        return inputs * score

# Load the model and tokenizer
with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
    model = load_model("models/sms_spam_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")

# Define the maximum sequence length
MAX_LEN = 100

# Define the input schema using Pydantic
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Define prediction endpoint
@app.post("/predict")
def predict(input_data: TextInput):
    try:
        # Preprocess the text
        processed_text = preprocess_text(input_data.text)
        text_seq = tokenizer.texts_to_sequences([processed_text])
        text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=MAX_LEN)
        
        # Make prediction
        prediction = model.predict(text_pad)
        result = "SPAM" if prediction > 0.6 else "NOT SPAM"
        
        # Return the result
        return {
            "text": input_data.text,
            "prediction": result,
            "probability": float(prediction[0][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)