
import nbformat as nbf
import os

with open('notebooks/sentiment_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Section 3: EDA
cells = []
cells.append(nbf.v4.new_markdown_cell('## 3. Exploratory Data Analysis (EDA)\nWe visualize the class balance, review lengths, and generate WordClouds to see the most frequent words.'))
cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df_sample, palette='viridis')
plt.title('Sentiment Distribution (Sample)')
plt.show()

# Review Length
df_sample['review_len'] = df_sample['cleaned_review'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df_sample[df_sample['sentiment']=='positive']['review_len'], color='green', label='Positive', kde=True)
sns.histplot(df_sample[df_sample['sentiment']=='negative']['review_len'], color='red', label='Negative', kde=True)
plt.title('Review Length Distribution')
plt.legend()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""
from wordcloud import WordCloud

def generate_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

pos_text = ' '.join(df_sample[df_sample['sentiment']=='positive']['cleaned_review'])
neg_text = ' '.join(df_sample[df_sample['sentiment']=='negative']['cleaned_review'])

generate_wordcloud(pos_text, 'Positive Reviews WordCloud')
generate_wordcloud(neg_text, 'Negative Reviews WordCloud')
"""))

nb.cells.extend(cells)
with open('notebooks/sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
