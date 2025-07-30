from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils_api import *
from src.grad_cam import make_prediction as make_gradcam
from PIL import Image
import numpy as np
import io
import cv2
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier_model = load_classifier_model()
segmentation_model = load_segmentation_model()
labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    img_input = preprocess_image_classifier(image_np, size=(224, 224))
    preds = classifier_model.predict(img_input)[0]
    label = labels[np.argmax(preds)]
    confidence = float(preds[np.argmax(preds)])

    if label == 'No Tumor':
        _, im_png = cv2.imencode('.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        overlay_bytes = im_png.tobytes()
    else:
        seg_input = preprocess_image_segmentation(image_np, size=(256, 256))
        mask = segmentation_model.predict(seg_input)[0]
        overlay = apply_mask(image_np, mask)
        _, im_png = cv2.imencode('.png', overlay)
        overlay_bytes = im_png.tobytes()

    return {
        "class_label": label,
        "confidence": confidence,
        "overlay_image": overlay_bytes.hex()
    }

@app.post("/gradcam/")
async def generate_gradcam(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    cam_path, _ = make_gradcam(image_np, classifier_model, last_conv_layer_name="conv5_block3_out")
    with open(cam_path, "rb") as f:
        cam_bytes = f.read()
    return {
        "gradcam_image": cam_bytes.hex()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
