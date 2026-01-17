from tensorflow import keras
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# ------------------------------------------------------------------
# CONFIGURABLE PARAMETERS
# ------------------------------------------------------------------
HSV_LOWER = np.array([ 0, 40,  60], dtype=np.uint8)   # tighter skin range
HSV_UPPER = np.array([20,150, 255], dtype=np.uint8)
SKIN_RATIO_THRESH = 0.20   # 20 % of pixels must be skin‑coloured
CONFIDENCE_THRESH  = 0.70   # 70 % soft‑max prob required for a label
MODEL_PATH = "output/skin_Model.h5"
LABELS = ["Acne", "Hair Loss", "Nail Fungus", "Normal", "Skin Allergy"]
# ------------------------------------------------------------------


def is_skin_present(img: np.ndarray,
                    lower=HSV_LOWER,
                    upper=HSV_UPPER,
                    ratio_thresh=SKIN_RATIO_THRESH) -> bool:
    """Return True if >= ratio_thresh of pixels are skin‑like in HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.count_nonzero(mask) / mask.size
    return skin_ratio >= ratio_thresh


def load_and_preprocess(img_path: str) -> np.ndarray:
    """Read an image, convert to RGB, resize to 224×224, normalise [0‑1]."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"[E] Could not read image: {img_path}")
    if not is_skin_present(bgr):
        sys.exit("[E] No significant skin detected. Try a clearer image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    arr = rgb.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    # ── CLI ─────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(description="Skin‑disease inference script")
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = ap.parse_args()

    if not Path(MODEL_PATH).is_file():
        sys.exit(f"[E] Model not found: {MODEL_PATH}")

    # ── Predict ─────────────────────────────────────────────────────
    model = keras.models.load_model(MODEL_PATH)
    x = load_and_preprocess(args.image)

    preds = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(preds))
    prob  = float(preds[idx])

    if prob < CONFIDENCE_THRESH:
        print(f"Result: Uncertain / None‑of‑the‑above  (confidence {prob:.2%})")
    else:
        print(f"Diagnosis: {LABELS[idx]}")
        print(f"Confidence: {prob:.2%}")


if __name__ == "__main__":
    main()
