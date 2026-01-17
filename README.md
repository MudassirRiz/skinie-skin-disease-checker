# Skinie – Skin Disease Checker

##  Project Overview

Skinie is an educational machine learning project built to explore how deep learning
models can be integrated into real-world web applications. The goal of the project
is to demonstrate end-to-end development, from model inference to user interaction,
using a simple and accessible interface.

##  How It Works

1. User uploads an image through the web interface.
2. The backend validates the image and performs preprocessing.
3. A VGG-19 based CNN model processes the image.
4. The predicted skin condition is generated.
5. The result is displayed on the UI.


##  Tech Stack

**Backend & ML**
- Python
- Flask
- TensorFlow
- Keras
- VGG-19 (CNN)

**Image Processing**
- OpenCV
- Pillow (PIL)

**Frontend**
- HTML
- CSS
- JavaScript

##  Screenshots of this ML Project

- Home Page  
- Image Upload Interface  
- Prediction Result  

*(Screenshots available in the repository)* 
PATH: "/static/Screenshots/"  (Working screenshots)


---

##  Features

| Feature                     | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| 🔍 **AI prediction**        | VGG‑19 CNN trained on labeled dermatology images               |
| 🖼 **Drag‑and‑drop upload** | User‑friendly image selection with live preview                |
| ⚡ **Skin‑presence guard**   | Quick OpenCV HSV skin‑pixel check before inference             |
| 🎨 **Custom UI**            | Poppins font, animated gradient background, particles.js layer |
| 📦 **Single‑file model**    | Pre‑trained `.h5` shipped in `/output` – no extra download     |

---

##  Project Structure

```text
Skinie/
├─ app.py                 # Flask entry point
├─ output/
│  └─ skin_Model.h5       # Saved VGG‑19 model
├─ templates/
│  └─ index.html          # Main UI template (Jinja2)
├─ static/
│  ├─ css/skinie.css      # Custom styles
│  ├─ uploads/            # (Runtime) uploaded images
│  └─ …                   # Fonts, icons, JS libraries
└─ README.md              # <‑‑ you are here
```

---

## 🛠Prerequisites

* **Python = 3.10** (tested on 3.10)
* **pip** 


---

🔧 Installation
```bash
# 1 – Create and activate a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

# 1 – Create and activate a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate       

# 3 – Install Python dependencies
$ pip install --upgrade pip         #use python version (3.10.**)
$ pip install -r requirements.txt 
$ pip install tensorflow 
$ pip install keras 
$ pip install flask 
$ pip install numpy
$ pip install openCv
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


# 4 – Run the app (dev server)
$ python app.py                  

# 5 – Visit
🔗 http://127.0.0.1:5000
```

### `requirements.txt` (minimal)

```text
Flask>=3.0
tensorflow>=2.15
numpy
opencv-python
Pillow
```

---

 Usage

1. Open the web page.
2. Fill in name, age, gender, affected area.
3. Drag‑and‑drop or click to upload a clear image (≤ 5 MB, JPG/PNG).
4. Press **Upload and Continue**.
5. The prediction card shows the result centered, bold, and large.

> ⚠️ **Disclaimer:** This tool is **not** a substitute for professional medical advice.

---


⚠️ Note: Once a result is predicted by the Model, refresh/reload the browser window/site for another prediction result.



 Retraining / Updating the Model

1. Collect images into `dataset/{class_name}/…`.
2. Adapt or create a `train.py` script (see sample under `/scripts`).
3. Train: `python train.py --epochs 25 --save output/skin_Model.h5`.
4. Restart the Flask app – it will automatically load the new model at launch.

---

 Troubleshooting

| Problem                                     | Fix                                                                    |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| *`ModuleNotFoundError: cv2`*                | `pip install opencv-python`                                            |
| *`ValueError: Unknown activation: 'relu6'`* | Upgrade/downgrade TensorFlow to match model version                    |
| Web page not loading CSS                    | Verify `<link … css/skinie.css>` path and `static_folder` in `Flask()` |
| GPU memory errors                           | Set `TF_FORCE_GPU_ALLOW_GROWTH=true` or use CPU TensorFlow             |

---



---


## Limitations

- This project is intended for educational purposes only.
- Prediction accuracy depends on image quality and lighting.
- Limited to a fixed set of skin disease categories.

 Authors & Acknowledgements

Mudassir Rizvi
 TensorFlow & Keras community for the deep‑learning stack
 Inspiration from open‑source healthcare UI kits

Have questions? Open an issue or reach me at [mdssrrizvi@gmail.com](mailto:mdssrrizvi@gmail.com).