import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os  # Import the os module
import sys  # Import the sys module for exiting the script

# Ensure the 'data/' directory exists
if not os.path.exists("data"):
    print("❌ Error: The 'data/' folder does not exist.")
    print("Please run the following scripts in order:")
    print("1. data_preprocessing.py")
    print("2. model_training.py")
    print("3. This script (plot_generation.py)")
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
    
    
# Calculate the number of spam and ham messages
label_counts = df['label'].value_counts()
num_ham = label_counts[0]  # Assuming 0 is the label for ham
num_spam = label_counts[1]  # Assuming 1 is the label for spam

print(f"Number of ham messages: {num_ham}")
print(f"Number of spam messages: {num_spam}")

# Create new features
df['countCharacters'] = df['message'].apply(len)
df['countWords'] = df['message'].apply(lambda i: len(nltk.word_tokenize(i)))
df['countSentences'] = df['message'].apply(lambda i: len(nltk.sent_tokenize(i)))

# Generate Word Clouds
spam_text = df[df['label'] == 1]['cleaned_message'].str.cat(sep=" ")
ham_text = df[df['label'] == 0]['cleaned_message'].str.cat(sep=" ")

spam_wc = WordCloud(width=500, height=500, background_color='white').generate(spam_text)
ham_wc = WordCloud(width=500, height=500, background_color='white').generate(ham_text)

plt.figure(figsize=(12, 6))
plt.imshow(spam_wc)
plt.axis("off")
plt.title("Word Cloud - Spam Messages")
plt.savefig("plots/wordcloud_spam.png")
plt.close()
print("✅ Word Cloud - Spam Message plotted and saved as plots/wordcloud_spam.png")

plt.figure(figsize=(12, 6))
plt.imshow(ham_wc)
plt.axis("off")
plt.title("Word Cloud - Non-Spam Messages")
plt.savefig("plots/wordcloud_ham.png")
plt.close()
print("✅ Word Cloud - Non-Spam Message plotted and saved as plots/wordcloud_ham.png")

# Generate Pairplot to Identify Relationship Between Features
sns.pairplot(df[['countCharacters', 'countWords', 'countSentences', 'label']], hue="label")
plt.savefig("plots/pairplot.png")
plt.close()
print("✅ Pairplot plotted and saved as plots/pairplot.png")

# Correlation Matrix and Heatmap
corr_matrix = df[['countCharacters', 'countWords', 'countSentences', 'label']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("plots/heatmap.png")
plt.close()
print("✅ Heatmap plotted and saved as plots/heatmap.png")

# Generate Pie Chart for the data distribution
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=['NOT SPAM', 'SPAM'], autopct='%0.2f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title("Distribution of Spam and Non-Spam Messages")
plt.savefig("plots/pie_chart.png")
plt.close()
print("✅ Pie Chart plotted and saved as plots/pie_chart.png")


print("✅✅ All plotting completed!")