# .\venv\Scripts\Activate.ps1  

from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


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

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detector.process(rgb)

    face_detected = results.detections is not None
    face_count = len(results.detections) if face_detected else 0

    response = {
        "success": True,
        "faceDetected": face_detected,
        "faces": face_count,
        "emotion": "unknown",
        "scores": {}
    }

    if face_detected:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        ih, iw = image.shape[:2]
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)

        x = max(0, x)
        y = max(0, y)
        w = max(1, min(iw - x, w))
        h = max(1, min(ih - y, h))

        face_roi = image[y : y + h, x : x + w]

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())

        lh = gray.shape[0]
        lower_third = gray[int(lh * 0.6) : lh, :]
        mouth_var = float(np.var(lower_third))

        upper_third = gray[0 : int(lh * 0.35), :]
        eye_var = float(np.var(upper_third))

        score = float(det.score[0]) if det.score else 0.0

        emotion = "neutral"
        if sharpness < 20 or eye_var < 200:
            emotion = "tired"
        elif mouth_var > 2000:
            emotion = "nervous"
        elif score > 0.7 and sharpness > 70 and mouth_var < 800:
            emotion = "confident"
        elif sharpness > 90 and mouth_var < 500:
            emotion = "stressed"

        response.update(
            {
                "emotion": emotion,
                "scores": {
                    "detectionScore": score,
                    "sharpness": sharpness,
                    "mouthVar": mouth_var,
                    "eyeVar": eye_var,
                },
            }
        )

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)