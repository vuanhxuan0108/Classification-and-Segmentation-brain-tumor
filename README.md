# 🧠 Brain Tumor Detection & Segmentation  

Hệ thống này thực hiện **phân loại** và **phân đoạn** khối u não từ ảnh MRI, ứng dụng **Deep Learning** và triển khai hoàn chỉnh qua API và giao diện web.  
- 🧪 **Phân loại** u não với mô hình **ResNet50**  
- 🧠 **Phân đoạn** khối u bằng **UNet (EfficientNetB0)**  
- 🔥 **Giải thích mô hình** bằng **Grad-CAM**  
- 🌐 **API** với **FastAPI** hỗ trợ chẩn đoán và quản lý bệnh án  
- 💻 **UI** với **Streamlit**  
- 📦 **Triển khai** bằng **Docker**  

## 📁 Mục lục
- [1. Giới thiệu](#1-giới-thiệu)  
- [2. API Hệ Thống](#2-api-hệ-thống)  
- [3. Kiến trúc mô hình](#3-kiến-trúc-mô-hình)  
- [4. Dữ liệu sử dụng](#4-dữ-liệu-sử-dụng)  
- [5. Hướng dẫn chạy](#5-hướng-dẫn-chạy)  
- [6. Đóng gói & Chạy với Docker](#6-đóng-gói--chạy-với-docker)  
- [7. Cấu trúc thư mục](#7-cấu-trúc-thư-mục)  
- [8. Giải thích mô hình (Grad-CAM)](#8-giải-thích-mô-hình-grad-cam)  
- [9. Kết quả](#9-kết-quả)  

## 1. Giới thiệu  
Phát hiện và phân đoạn khối u não từ ảnh MRI là một bài toán quan trọng trong y tế. Hệ thống này:  
- ✅ Phân loại ảnh MRI có khối u hay không  
- ✅ Phân đoạn vùng khối u chính xác  
- ✅ Trực quan hóa vùng mô hình quan tâm bằng Grad-CAM  
- ✅ Lưu trữ & quản lý bệnh án bệnh nhân  

## 2. API Hệ Thống  
API được xây dựng bằng **FastAPI**, hỗ trợ:  
- 📤 Upload nhiều ảnh MRI để phân tích cùng lúc  
- 🔍 Trả về nhãn, độ tin cậy và vùng phân đoạn khối u  
- 🔥 Cung cấp Grad-CAM để giải thích mô hình  
- 🧾 Lưu và quản lý thông tin bệnh nhân, lịch sử chẩn đoán  
- 📦 Xuất toàn bộ ảnh kết quả (.zip) và báo cáo chẩn đoán (.pdf)  

## 3. Kiến trúc mô hình  
- **Phân loại**:  
  - `ResNet50` với Loss `CategoricalCrossentropy`  
  - Metrics: `Accuracy`, `F1-score`  
- **Phân đoạn**:  
  - `UNet` (backbone `EfficientNetB0`)  
  - Loss: `Tversky loss`  
  - Metrics: `IoU`, `Tversky index`  

## 4. Dữ liệu sử dụng  
- 📂 **Phân loại**: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- 📂 **Phân đoạn**: [Brain Tumor Segmentation (BraTS)](https://www.kaggle.com/datasets/anhxunv/brain-turmor-segment-datasets)  

## 5. Hướng dẫn chạy  
### ✅ Cài đặt thư viện  
```bash
pip install -r requirements.txt
```
### 🚀 Huấn luyện mô hình  
```bash
notebook/brain-tumor-classification-resnet50.ipynb
notebook/brain-tumor-segmentation-unet.ipynb
```
### 🌐 Chạy API  
```bash
python app/main_api.py
```
👉 Truy cập: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
### 💻 Chạy giao diện Streamlit  
```bash
streamlit run app/app_streamlit.py
```
👉 Truy cập: [http://localhost:8501](http://localhost:8501)  

## 6. Đóng gói & Chạy với Docker  
### 🛠️ Build image:  
```bash
docker build -t brain-tumor-app .
```
### 🚀 Chạy container:  
```bash
docker run -d -p 8000:8000 -p 8501:8501 brain-tumor-app
```
### 🐳 Hoặc dùng docker-compose:  
```bash
docker-compose up --build -d
```
👉 Sau khi chạy:  
- FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)  
- Streamlit: [http://localhost:8501](http://localhost:8501)  

## 7. Cấu trúc thư mục  
```
.
├── app/
│   ├── app_streamlit.py
│   ├── main_api.py
│   ├── model_definitions.py
│   └── utils_api.py
├── model_save/
├── notebook/
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── start.sh
└── README.md
```

## 8. Giải thích mô hình (Grad-CAM)  
💡 Hệ thống tích hợp **Grad-CAM** để trực quan hóa vùng ảnh MRI mà mô hình ResNet50 sử dụng:  
- 🎯 Giúp bác sĩ hiểu **tại sao mô hình dự đoán như vậy**  
- 🔥 UI Streamlit cho phép bật/tắt **Grad-CAM overlay**  

## 9. Kết quả  
| Tác vụ       | Mô hình               | Accuracy / IoU | Ghi chú |
|--------------|------------------------|----------------|---------|
| Phân loại    | ResNet50               | ~98% Accuracy  | Hỗ trợ XAI (Grad-CAM) |
| Phân đoạn    | UNet (EfficientNetB0) | ~0.91 Tversky, ~0.81 IoU |         |

