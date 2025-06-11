
# 🧠 Brain Tumor Detection & Segmentation

Dự án này thực hiện **phân loại** và **phân đoạn** khối u não từ ảnh MRI, ứng dụng các kỹ thuật Deep Learning hiện đại:
- Phân loại u não với mô hình **ResNet50**
- Phân đoạn khối u bằng **UNet++ (backbone EfficientNetB3)**
- Tích hợp mô hình vào hệ thống API bằng **FastAPI**
- Giao diện demo bằng **Streamlit**

---

## 📁 Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Kiến trúc mô hình](#2-kiến-trúc-mô-hình)
- [3. Dữ liệu sử dụng](#3-dữ-liệu-sử-dụng)
- [4. Hướng dẫn chạy](#4-hướng-dẫn-chạy)
- [5. Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
- [6. Kết quả](#6-kết-quả)

---

## 1. Giới thiệu

Phát hiện và phân đoạn u não từ ảnh MRI là một bài toán quan trọng trong lĩnh vực chẩn đoán hình ảnh y tế. Dự án này giúp:
- Phân loại ảnh MRI có chứa u não hay không.
- Nếu có, phân đoạn chính xác vị trí khối u.

---

## 2. Kiến trúc mô hình

- **Phân loại**:
  - Mô hình: `ResNet50`
  - Loss: `CategoricalCrossentropy`
  - Metrics: `Accuracy`, `F1-score`
- **Phân đoạn**:
  - Mô hình: `UNet++` với `EfficientNetB3` làm backbone
  - Loss: `Tversky loss`
  - Metrics: `IoU`, `tversky`

---

## 3. Dữ liệu sử dụng

### 📂 Phân loại (Classification)
- Dataset: **Brain Tumor Classification (MRI)**
- Link: [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### 📂 Phân đoạn (Segmentation)
- Dataset: **Brain Tumor Segmentation (BraTS)**
- Link: [Kaggle Dataset](https://www.kaggle.com/datasets/anhxunv/brain-turmor-segment-datasets)

---

## 4. Hướng dẫn chạy

### ✅ Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 🚀 Huấn luyện mô hình
**Phân loại:**
```bash
# notebook
notebook/brain-tumor-classification-resnet50.ipynb
```

**Phân đoạn:**
```bash
# notebook
notebook/brain-tumor-segmentation-unet.ipynb
```

### 🌐 Chạy API với FastAPI
```bash
uvicorn main_api:app --reload
```

Mở trình duyệt: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 💻 Chạy giao diện Streamlit
```bash
streamlit run app_streamlit.py
```

---

## 5. Cấu trúc thư mục

```
.
├── code_py/
│   ├── grad_cam.py
│   ├── load_datasets.py
│   ├── preprocessing.py
│   └── utils.py
├── image_model/
│   ├── resnet_kiank.png
│   └── ...
├── model_save/
│   ├── best_model_resnet50.keras
│   └── best_model_efficient_unet.keras
├── notebook/
│   ├── brain-tumor-classification-resnet50.ipynb
│   └── brain-tumor-segmentation-unet.ipynb
├── app_streamlit.py
├── main_api.py
├── model_definitions.py
├── utils_api.py
├── requirements.txt
└── README.md
```

---

## 6. Kết quả

| Tác vụ       | Mô hình               | Accuracy / IoU | Ghi chú |
|--------------|------------------------|----------------|---------|
| Phân loại    | ResNet50               | ~98% Accuracy  |         |
| Phân đoạn    | UNet++ (EfficientNetB3)| ~0.91 Tversky, ~0.82 IoU  |         |

---
