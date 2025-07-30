import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os
from tqdm import tqdm

def plot_image_comparison(before, after, title, cmap=None):
    plt.figure(figsize=(8, 4))
    plt.suptitle(title, fontsize=14)
    plt.subplot(1, 2, 1)
    plt.title("Before")
    plt.imshow(before, cmap=cmap)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("After")
    plt.imshow(after, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def estimate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def unsharp_mask(img, blur_ksize=(5,5), strength=3.0, sharpness_thresh = 80, plot=False):
    sharpness = estimate_sharpness(img)
    if sharpness < sharpness_thresh:
        blurred = cv2.GaussianBlur(img, blur_ksize, 0)
        sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    else:
        sharpened = img.copy()

    if plot:
        plot_image_comparison(img, sharpened, f"Unsharp Mask (sharpness={sharpness:.1f})")
    
    return sharpened

def brighten_if_dark(img, brightness_thresh=40, gamma=1.3, plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < brightness_thresh:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        brightened = cv2.LUT(img, table)
    else:
        brightened = img.copy()

    if plot:
        plot_image_comparison(img, brightened, f"Gamma Brightening (brightness={mean_brightness:.1f})")

    return brightened


def skull_strip(img, plot=False, re_crop=False, thresh_blur = 20):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, thresh_blur, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    ctns = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = ctns
    if not contours:
        return img
    
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    
    if plot:
        plot_image_comparison(img, img_masked, "Skull Stripping")
    
    if re_crop:
        return ctns
    else:
        return img_masked


def crop_imgs(img, plot=False, thresh_blur = 40):
    cnts = skull_strip(img, re_crop=True, thresh_blur=thresh_blur)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)
    
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)
    
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    new_img = cv2.resize(
                new_img,
                dsize=(224, 224),
                interpolation=cv2.INTER_CUBIC
            )


    if plot:
        plt.figure(figsize=(15,6))
        plt.subplot(141)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Step 1. Get the original image')
        plt.subplot(142)
        plt.imshow(img_cnt)
        plt.xticks([])
        plt.yticks([])
        plt.title('Step 2. Find the biggest contour')
        plt.subplot(143)
        plt.imshow(img_pnt)
        plt.xticks([])
        plt.yticks([])
        plt.title('Step 3. Find the extreme points')
        plt.subplot(144)
        plt.imshow(new_img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Step 4. Crop the image')
        plt.show()
    return new_img

def apply_soft_clahe(img, std_thresh=60, plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.std(gray) >= std_thresh:
        if plot:
            print(f"CLAHE skipped (std={np.std(gray):.1f})")
        return img
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    clahe_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    if plot:
        plot_image_comparison(img, clahe_img, f"CLAHE (std={np.std(gray):.1f})")
    
    return clahe_img


def process_and_save_images_classification(input_dir, output_dir, splits=["Training", "Testing"], brightness_thresh=40, std_thresh=60, sharpness_thresh=80, thresh_blur = 20, strength=3.0):
    for split in splits:
        for cls in os.listdir(os.path.join(input_dir, split)):
            input_class_dir = os.path.join(input_dir, split, cls)
            output_class_dir = os.path.join(output_dir, split, cls)
            os.makedirs(output_class_dir, exist_ok=True)
            
            for img_name in tqdm(os.listdir(input_class_dir), desc=f"{split}/{cls}"):
                try:
                    img_path = os.path.join(input_class_dir, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = unsharp_mask(img, sharpness_thresh=sharpness_thresh, strength=strength)
                    img = brighten_if_dark(img, brightness_thresh=brightness_thresh)
                    img = apply_soft_clahe(img, std_thresh=std_thresh)
                    img = skull_strip(img, thresh_blur=thresh_blur)
                    img = crop_imgs(img, thresh_blur=thresh_blur)
                    
                    save_path = os.path.join(output_class_dir, img_name)
                    cropped_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, cropped_img_bgr)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

                    

def process_and_save_images_and_masks_segmentation(image_dir, mask_dir, out_dir, brightness_thresh=40, std_thresh=60, sharpness_thresh=80, thresh_blur=30, strength=3.0):
    out_image_dir = os.path.join(out_dir, "image")
    out_mask_dir = os.path.join(out_dir, "mask")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(image_dir), desc="Processing"):
        try:
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"Error reading {img_name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = unsharp_mask(img, sharpness_thresh=sharpness_thresh, strength=strength)
            img = brighten_if_dark(img, brightness_thresh=brightness_thresh)
            img = apply_soft_clahe(img, std_thresh=std_thresh)
            img = skull_strip(img, thresh_blur=thresh_blur)

            out_img_path = os.path.join(out_image_dir, img_name)
            out_mask_path = os.path.join(out_mask_dir, img_name)
            cv2.imwrite(out_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, mask)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")