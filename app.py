from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename          
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)

# --------- Conffig and model --....
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('output/skin_Model.h5')
class_names = ['Acne', 'Hairloss', 'Nail Fungus', 'Normal', 'Skin Allergy']

# --------- helper:skin‑present? ---
def is_skin_present(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    skin_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
    return skin_ratio > 0.01           # ≥ 1 % skin pixels(ye check krta hai)

# --------- Routes - ,yahhi hai 
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction   = None
    image_url    = None   

    if request.method == 'POST':
        uploaded_file = request.files.get('img')  # <input name="img">
        if uploaded_file and uploaded_file.filename:
            # ---- Save the file ----------------------------------------------------
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)
            image_url = url_for('static', filename=f'uploads/{filename}')

            # ---- Optional skin check -
            if not is_skin_present(filepath):
                prediction = "Skin not detected in image."
                return render_template('index.html',
                                       prediction=prediction,
                                       image_url=image_url)

            # ---- model prediction --
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            result     = model.predict(img_array)
            prediction = class_names[np.argmax(result)]

            # Return page with image and result
            return render_template('index.html',
                                   prediction=prediction,
                                   image_url=image_url)

    # GET request (initial load)
    return render_template('index.html', prediction=prediction, image_url=image_url)


@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/dev')
def dev():
    return render_template('dev.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
