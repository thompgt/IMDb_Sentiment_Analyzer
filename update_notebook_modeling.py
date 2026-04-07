
import nbformat as nbf
import os

with open('notebooks/sentiment_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Section 4: Modeling
cells = []
cells.append(nbf.v4.new_markdown_cell('## 4. Modeling & Evaluation\nWe split the data, vectorize the text using TF-IDF, and train two models: Naive Bayes and Logistic Regression.'))
cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Encode Labels
df_sample['label'] = df_sample['sentiment'].map({'positive': 1, 'negative': 0})

X = df_sample['cleaned_review']
y = df_sample['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f'TF-IDF features: {X_train_tfidf.shape[1]}')
"""))

cells.append(nbf.v4.new_code_cell("""
# Model 1: Naive Bayes (Baseline)
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

print('Naive Bayes Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}')
print(classification_report(y_test, y_pred_nb))
"""))

cells.append(nbf.v4.new_code_cell("""
# Model 2: Logistic Regression (Improved)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

print('Logistic Regression Results:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')
print(classification_report(y_test, y_pred_lr))
"""))

cells.append(nbf.v4.new_code_cell("""
# Visualizing Results: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# ROC Curve
y_prob_lr = lr_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
"""))

nb.cells.extend(cells)
with open('notebooks/sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
