import pandas as pd
import os
import urllib.request

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Hugging Face dataset URLs 
DATASETS = {
    "dataset.csv": "https://huggingface.co/datasets/subashdhamee/sms-spam-ham-dataset/resolve/main/dataset.csv",
    "enron_spam_data.csv": "https://huggingface.co/datasets/subashdhamee/sms-spam-ham-dataset/resolve/main/enron_spam_data.csv",
}

# Define a better stopword list
custom_stopwords = set(stopwords.words('english')) - {"no", "not", "won't", "don't", "urgent", "free", "win", "claim", "offer"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def download_dataset(filename, url):
    """Downloads dataset if it doesn't exist."""
    save_path = os.path.join("data", filename)

    if os.path.exists(save_path):
        print(f"✅ {filename} already exists. Skipping download.")
    else:
        print(f"⬇ Downloading {filename}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ {filename} downloaded and saved to {save_path}.")

# Download all datasets
for file, url in DATASETS.items():
    download_dataset(file, url)

print("✅ All datasets downloaded successfully!")

def load_and_preprocess_data():
    """Loads datasets, preprocesses text, and merges them into one."""
    
    # Load datasets (FIXED Delimiters & Encoding Issues)
    df1 = pd.read_csv(
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        sep='\t', names=['label', 'message']
    )

    df2 = pd.read_csv("data/dataset.csv", encoding='utf-8', low_memory=False)
    
    df3 = pd.read_csv("data/enron_spam_data.csv", 
                      dtype={'Spam/Ham': str, 'Message': str}, 
                      low_memory=False, encoding='utf-8')

    print("✅ All datasets loaded successfully!")

    # Preprocess datasets
    df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})
    # #column 2, 3, 4 have majority missing values, so it is better to drop them.(Only while using the original csv from diffrent sites)
    # df1.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)

    # Rename columns to match the original dataset
    df2.rename(columns={'text_type': 'label', 'text': 'message'}, inplace=True)

    # Map labels to 0 (ham) and 1 (spam)
    df2['label'] = df2['label'].map({'ham': 0, 'spam': 1})
    
    
    # Drop unnecessary & unnamed columns
    df3 = df3.loc[:, ~df3.columns.str.contains('Unnamed')]
    df3 = df3.drop(['Message ID', 'Subject', 'Date'], axis=1)

    # Rename columns
    df3.rename(columns={'Spam/Ham': 'label', 'Message': 'message'}, inplace=True)

    # Remove leading/trailing spaces & drop NaN values from label column
    df3['label'] = df3['label'].astype(str).str.strip()
    df3 = df3.dropna(subset=['label'])

    # Convert cells to a single line
    df3['message'] = df3['message'].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

    # Keep only valid labels ("spam" or "ham")
    df3 = df3[df3['label'].isin(['spam', 'ham'])].copy()  # Use .copy() to prevent warnings
    # Map 'ham' to 0 and 'spam' to 1
    df3['label'] = df3['label'].map({'ham': 0, 'spam': 1})

    # Convert the 'label' column to int type
    df3['label'] = df3['label'].astype(int)

    # Drop NaN values in "message" column
    df3 = df3.dropna(subset=['message'])

    # Reset index after dropping rows
    df3.reset_index(drop=True, inplace=True)

    # Reorder columns
    df3 = df3[['label', 'message']]

    # Merge datasets
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df = df.drop_duplicates(keep='first')
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    
    # print(df[df['label'] == 1]['cleaned_message'].head(10))
    
    df = df.dropna(subset=['cleaned_message'])
    df['cleaned_message'] = df['cleaned_message'].replace('', 'empty_message')

    # Save merged dataset
    merged_dataset_path = "data/merged_spam_ham.csv"
    df.to_csv(merged_dataset_path, index=False, encoding='utf-8')
    print(f"✅ Merged dataset saved to {merged_dataset_path}")

    return df

# Run the function to ensure data is preprocessed
df = load_and_preprocess_data()
print("✅ Data preprocessing complete!")
