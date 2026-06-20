import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Step 1: Reading your new expanded dataset...")
# Because this script runs directly inside 'backend', it reads the file perfectly
df = pd.read_csv("train_expanded.csv")

# Automatically detect whatever your text and label column names are
text_col = max(df.columns, key=lambda col: df[col].astype(str).str.len().mean())
label_col = next((col for col in df.columns if col != text_col), df.columns[-1])

X = df[text_col].astype(str)
y = df[label_col].astype(int)

print("Step 2: Training your machine learning model brains...")
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

print("Step 3: Overwriting your old model files with the smart versions...")
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🎉 SUCCESS! Your new model is successfully trained for life!")