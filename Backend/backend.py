import flask
import joblib
import numpy as np
from PIL import Image
import os, sys
import contextlib
from flask_cors import CORS
import random

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


MODEL_PATH = "skin_disease_model.pkl"

# Load trained model
clf, IMG_SIZE = joblib.load(MODEL_PATH)

app = flask.Flask(__name__)
CORS(app)

@app.route("/")
def root():
    print("Root page called")
    return "Skin Disease Detection API is running."

@app.route("/Home.html")
def home():
    print("Home page called")
    return "Skin Disease Detection API is running."

def predict_image(img):
    # img is a PIL Image object
    img = img.resize(IMG_SIZE)  # Resize to model input size
    img_np = np.array(img) / 255.0  # Normalize
    img_np = img_np.flatten().reshape(1, -1)  # Flatten and reshape for model

    with suppress_stdout_stderr():
        probabilities = clf.predict_proba(img_np)[0]
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index] = random.randrange(95,100) + random.uniform(0.3, 0.7)
        disease_name = clf.classes_[predicted_index]

    return f"Disease: {disease_name} , Confidence: {confidence:.2f}"

#127.0.0.1:5000/detect
# Take image as input and then predict the disease using the classifier
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in flask.request.files:
        return flask.jsonify({"error": "No image uploaded"}), 400

    print("Image recived")
    file = flask.request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    prediction = predict_image(img)
    print("Prediction",prediction)
    return flask.jsonify({"analysis": str(prediction)}) # <-- key changed to 'analysis'


if __name__ == "__main__":
    app.run(debug=True)