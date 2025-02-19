import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load preprocessed data
df = pd.read_csv("data/merged_spam_ham.csv")
print("✅ Dataset import completed !")

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
print("✅ Word Cloud - Spam Message ploatted ad saved as plots/wordcloud_spam.png")

plt.figure(figsize=(12, 6))
plt.imshow(ham_wc)
plt.axis("off")
plt.title("Word Cloud - Non-Spam Messages")
plt.savefig("plots/wordcloud_ham.png")
plt.close()
print("✅ Word Cloud - Non-Spam Message ploatted ad saved as plots/wordcloud_ham.png")


# Generate Pairplot to Identify Relationship Between Features
sns.pairplot(df[['countCharacters', 'countWords', 'countSentences', 'label']], hue="label")
plt.savefig("plots/pairplot.png")
plt.close()
print("✅ Pair ploatted and saved as plots/pairplot.png ")


# Correlation Matrix and Heatmap
corr_matrix = df[['countCharacters', 'countWords', 'countSentences', 'label']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("plots/heatmap.png")
plt.close()
print("✅ Heatmap ploatted and saved as plots/heatmap.png")

# Generate Pie Chart for the data distribution 
plt.figure(figsize=(8, 8))
plt.pie(df['label'].value_counts(), labels=['NOT SPAM', 'SPAM'], autopct='%0.2f%%', radius=0.8)
plt.title("Distribution of Spam and Non-Spam Messages")
plt.savefig("plots/pie_chart.png")
plt.close()
print("✅ Pi-Chart ploatted and saved as plots/pie_chart.png")

print("✅✅ All ploatting completed!")