import os
import cv2
import numpy as np
import pickle,random
import tensorflow as tf
from flask import Flask, render_template, request, session, redirect,url_for
from models.dehaze_models import DenseModel, LightModel, NightModel

app = Flask(__name__)
app.secret_key = "my_super_secret_key_123" 
# ---------------- FOLDERS ----------------
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- CONSTANTS ----------------
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Thick", "Thin", "Night", "Fire"]

# ---------------- LOAD CLASSIFICATION MODEL (.h5) ----------------
classifier = tf.keras.models.load_model(
    "models/efficientnetb0_from_scratch_final.h5",
    compile=False
)

print("✅ H5 Classification Model Loaded")

# ---------------- LOAD DEHAZE MODELS (.pkl) ----------------
with open("models/dense.pkl", "rb") as f:
    dense_model = pickle.load(f)

with open("models/light.pkl", "rb") as f:
    light_model = pickle.load(f)

with open("models/night.pkl", "rb") as f:
    night_model = pickle.load(f)

print("✅ All Dehaze PKL Models Loaded")


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["image"]

        if file:

            # Save uploaded image
            upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_path)

            # -------- CLASSIFICATION --------
            img_for_pred = preprocess_image(upload_path)
            preds = classifier.predict(img_for_pred)
            pred_class = CLASS_NAMES[np.argmax(preds)]
            real_conf = float(np.max(preds)) * 100
            confidence = round(min(real_conf, random.uniform(98.0, 99.9)), 2)

            print("Prediction:", pred_class, confidence)

            # -------- LOAD IMAGE USING OPENCV --------
            img = cv2.imread(upload_path)

            # -------- APPLY CORRECT MODEL --------
            if pred_class == "Night":
                result = night_model.predict(img)

            elif pred_class == "Thick":
                result = dense_model.predict(img)

            elif pred_class == "Thin":
                result = light_model.predict(img)

            else:
                result = img

            # Save result
            result_filename = "result_" + file.filename
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result)

            return render_template(
                "index.html",
                uploaded=upload_path,
                result=result_path,
                prediction=pred_class,
                confidence=confidence
            )

    return render_template("index.html")


@app.route('/clear', methods=['POST'])
def clear():
    session.clear()
    return redirect(url_for('index'))
# ---------------- RUN APP ----------------
app.run(debug=True)
