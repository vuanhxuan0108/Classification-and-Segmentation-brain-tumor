import cv2
import numpy as np
from model_definitions import build_classifier_model, build_segmentation_model
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from code_py.preprocessing import *
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
preprocess_input = sm.get_preprocessing('efficientnetb1')

def load_classifier_model(model_path='model_save/best_model_resnet50.keras'):
    model = build_classifier_model()
    model.load_weights(model_path)
    return model

def load_segmentation_model(model_path='model_save/best_model_seg.keras'):
    model = build_segmentation_model()
    model.load_weights(model_path)
    return model

def preprocess_image_classifier(img_array, size=(224, 224)):
    img = cv2.resize(img_array, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = unsharp_mask(img, sharpness_thresh=80, strength=3.0)
    img = brighten_if_dark(img, brightness_thresh=40)
    img = apply_soft_clahe(img, std_thresh=60)
    img = skull_strip(img, thresh_blur=20)
    img = crop_imgs(img, thresh_blur=20)
    return np.expand_dims(img, axis=0)

def preprocess_image_segmentation(img_array, size=(224, 224)):
    img = cv2.resize(img_array, size)
    img = unsharp_mask(img, sharpness_thresh=80, strength=3.0)
    img = brighten_if_dark(img, brightness_thresh=40)
    img = apply_soft_clahe(img, std_thresh=60)
    img = skull_strip(img, thresh_blur=20)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def apply_mask(img, mask):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
    return overlay
