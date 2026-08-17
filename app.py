from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import urllib.request

app = Flask(__name__)

# ---------- Face detector (bbox only, bundled with opencv) ----------
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# ---------- Emotion model (ONNX FER+, real trained classifier) ----------
MODEL_PATH = "emotion-ferplus-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

EMOTIONS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading emotion model (~34MB, one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


ensure_model()
emotion_net = cv2.dnn.readNetFromONNX(MODEL_PATH)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_emotion(face_gray_roi):
    face_resized = cv2.resize(face_gray_roi, (64, 64))
    blob = face_resized.astype(np.float32).reshape(1, 1, 64, 64)

    emotion_net.setInput(blob)
    output = emotion_net.forward()[0]

    probs = softmax(output)
    top_idx = int(np.argmax(probs))

    return EMOTIONS[top_idx], {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "emotion-detection-api"}), 200


@app.route("/detect", methods=["POST"])
def detect_face():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        }), 400

    file = request.files["image"]
    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({
            "success": False,
            "message": "Invalid image"
        }), 400

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_full,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    face_detected = len(faces) > 0
    face_count = len(faces)

    response = {
        "success": True,
        "faceDetected": face_detected,
        "faces": face_count,
        "emotion": "unknown",
        "scores": {}
    }

    if face_detected:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        ih, iw = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(iw - x, w))
        h = max(1, min(ih - y, h))

        face_gray_roi = gray_full[y : y + h, x : x + w]

        emotion, all_scores = predict_emotion(face_gray_roi)

        response.update({
            "emotion": emotion,
            "scores": all_scores
        })

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)