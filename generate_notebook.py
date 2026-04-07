
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cells
cells = []

# Title & Description
cells.append(nbf.v4.new_markdown_cell('# SentimentSense: IMDb Movie Review Classifier 🎬\n\nWelcome to the SentimentSense project! In this notebook, we build a complete pipeline to classify movie reviews as positive or negative using the IMDb dataset.'))

# Section 1: Setup
cells.append(nbf.v4.new_markdown_cell('## 1. Setup and Data Loading\nFirst, we install the necessary libraries and download our dataset from Kaggle.'))
cells.append(nbf.v4.new_code_cell('import os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport re\nimport nltk\nfrom nltk.corpus import stopwords\nfrom nltk.stem import WordNetLemmatizer\nimport kagglehub\n\n# Download NLTK data\nnltk.download("stopwords")\nnltk.download("wordnet")\nnltk.download("omw-1.4")'))

# Data download logic
cells.append(nbf.v4.new_code_cell('# Download dataset using kagglehub\n# Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews\npath = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")\nprint("Path to dataset files:", path)\n\n# Check for CSV in the download path\nfiles = os.listdir(path)\ncsv_file = [f for f in files if f.endswith(".csv")][0]\ncsv_path = os.path.join(path, csv_file)\ndf = pd.read_csv(csv_path)\n\nprint(f"Dataset loaded with {len(df)} reviews.")\ndf.head()'))

# Section 2: Preprocessing
cells.append(nbf.v4.new_markdown_cell('## 2. Text Preprocessing\n\nWe clean the text by:\n1. Removing HTML tags.\n2. Lowercasing.\n3. Removing special characters and punctuation.\n4. Removing stopwords.\n5. Applying Lemmatization.'))
cells.append(nbf.v4.new_code_cell("""
from tqdm import tqdm
tqdm.pandas()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Tokenize and remove stopwords + lemmatize
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

# For speed during demonstration, we sample 5000 reviews
df_sample = df.sample(5000, random_state=42).reset_index(drop=True)
print("Cleaning reviews...")
df_sample['cleaned_review'] = df_sample['review'].progress_apply(clean_text)
df_sample.head()
"""))

nb.cells = cells
with open('notebooks/sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
