import pandas as pd
import os
import urllib.request

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Hugging Face dataset URLs (FIXED URLs)
DATASETS = {
    "dataset.csv": "https://huggingface.co/datasets/subashdhamee/sms-spam-ham-dataset/resolve/main/dataset.csv",
    "enron_spam_data.csv": "https://huggingface.co/datasets/subashdhamee/sms-spam-ham-dataset/resolve/main/enron_spam_data.csv",
}

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

    df2.rename(columns={'text_type': 'label', 'text': 'message'}, inplace=True)
    df2['label'] = df2['label'].map({'ham': 0, 'spam': 1})
    
    df3 = df3.loc[:, ~df3.columns.str.contains('Unnamed')]
    df3 = df3.drop(columns=['Message ID', 'Subject', 'Date'], errors='ignore')
    df3.rename(columns={'Spam/Ham': 'label', 'Message': 'message'}, inplace=True)
    # Convert cells to a single line
    df3['message'] = df3['message'].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    df3 = df3[df3['label'].isin(['spam', 'ham'])].copy()
    df3['label'] = df3['label'].map({'ham': 0, 'spam': 1})
    df3['label'] = df3['label'].astype(int)
    
    df3 = df3.dropna(subset=['message'])
    df3.reset_index(drop=True, inplace=True)
    df3 = df3[['label', 'message']]

    # Merge datasets
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df = df.drop_duplicates(keep='first')

    # Save merged dataset
    merged_dataset_path = "data/merged_spam_ham.csv"
    df.to_csv(merged_dataset_path, index=False, encoding='utf-8')
    print(f"✅ Merged dataset saved to {merged_dataset_path}")

    return df

# Run the function to ensure data is preprocessed
df = load_and_preprocess_data()
print("✅ Data preprocessing complete!")
