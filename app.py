import os
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "blaze_face_short_range.tflite"
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)

face_detector = vision.FaceDetector.create_from_options(
    options
)

@app.route("/detect", methods=["POST"])
def detect_face():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        }), 400


    file = request.files["image"]


    data = np.frombuffer(
        file.read(),
        np.uint8
    )


    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


    if image is None:
        return jsonify({
            "success": False,
            "message": "Invalid image"
        }), 400


    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    results = face_detector.detect(mp_image)


    detections = results.detections

    face_detected = len(detections) > 0
    face_count = len(detections)


    response = {
        "success": True,
        "faceDetected": face_detected,
        "faces": face_count,
        "emotion": "unknown",
        "scores": {}
    }



    if not face_detected:
        return jsonify(response)



    det = detections[0]


    bbox = det.bounding_box


    ih, iw = image.shape[:2]


    x = int(bbox.origin_x)
    y = int(bbox.origin_y)

    w = int(bbox.width)
    h = int(bbox.height)

    x = max(0, x)
    y = max(0, y)

    w = max(
        1,
        min(iw - x, w)
    )

    h = max(
        1,
        min(ih - y, h)
    )


    face_roi = image[
        y:y + h,
        x:x + w
    ]


    if face_roi.size == 0:
        return jsonify({
            "success": True,
            "faceDetected": face_detected,
            "faces": face_count,
            "emotion": "unknown",
            "scores": {}
        })



    gray = cv2.cvtColor(
        face_roi,
        cv2.COLOR_BGR2GRAY
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        lap.var()
    )


    lh = gray.shape[0]

    lower_third = gray[
        int(lh * 0.6):lh,
        :
    ]

    mouth_var = float(
        np.var(lower_third)
    )


    upper_third = gray[
        0:int(lh * 0.35),
        :
    ]

    eye_var = float(
        np.var(upper_third)
    )


    score = 0.0

    if det.categories:
        score = float(
            det.categories[0].score
        )



    emotion = "neutral"


    if sharpness < 20 or eye_var < 200:

        emotion = "tired"

    elif mouth_var > 2000:

        emotion = "nervous"

    elif (
        score > 0.7
        and sharpness > 70
        and mouth_var < 800
    ):

        emotion = "confident"

    elif (
        sharpness > 90
        and mouth_var < 500
    ):

        emotion = "stressed"

    response.update({

        "emotion": emotion,

        "scores": {

            "detectionScore": score,

            "sharpness": sharpness,

            "mouthVar": mouth_var,

            "eyeVar": eye_var

        }

    })


    return jsonify(response)



if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )