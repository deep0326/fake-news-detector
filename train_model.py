import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
fake = pd.read_csv("News_Dataset/Fake.csv")
true = pd.read_csv("News_Dataset/True.csv")

# 2. Add labels (0 = Fake, 1 = True)
fake["label"] = 0
true["label"] = 1

# 3. Combine and shuffle
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)

# 4. Use title + text as input
df["content"] = df["title"] + " " + df["text"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# 6. Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 8. Check accuracy
predictions = model.predict(X_test_vec)
print(f"✅ Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 9. Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model saved!")