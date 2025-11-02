from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# === Configuration ===
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load Pre-trained Model ===
MODEL_PATH = "emotion_detection_model.h5"
model = load_model(MODEL_PATH)

# === Initialize Database ===
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS uploads
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     emotion TEXT,
                     upload_time TEXT)''')
    conn.close()

init_db()

# === Emotion Labels (FER2013) ===
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


@app.route('/')
def index():
    """Render the upload page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle uploaded image file."""
    file = request.files.get('image')
    if not file:
        return render_template('index.html', emotion="⚠️ No image uploaded.")

    # Save file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    prediction = model.predict(img, verbose=0)
    emotion = EMOTIONS[np.argmax(prediction)]

    # Save record in database
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO uploads (filename, emotion, upload_time) VALUES (?, ?, ?)",
                (file.filename, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template('index.html', filename=file.filename, emotion=emotion)


if __name__ == '__main__':
    app.run(debug=True)
    