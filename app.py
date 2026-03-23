from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]
    transformed = vectorizer.transform([news_text])
    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed)[0]
    score = round(max(confidence) * 100, 2)
    
    if prediction == 1:
        result = f"✅ REAL News — {score}% confident"
    else:
        result = f"❌ FAKE News — {score}% confident"
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)