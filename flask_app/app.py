from flask import Flask,render_template,request
import preprocessing_utility as ppu
import mlflow.pyfunc
import pickle
import pandas as pd

model = mlflow.pyfunc.load_model("models:/MyRegisterModel@production")
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form['text']

    # clean
    text = ppu.normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(
        features.toarray(), columns=[str(i) for i in range(features.shape[1])]
    )

    # prediction
    result = model.predict(features_df)
    return render_template('index.html', result=result[0])


app.run(debug=True, host="127.0.0.2", port=5002)
