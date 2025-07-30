import streamlit as st
import requests
from PIL import Image
import io
import zipfile
import os
from fpdf import FPDF

st.set_page_config(page_title="Brain Tumor Analyzer", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "intro"
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}
if "results" not in st.session_state:
    st.session_state.results = []

if st.session_state.page == "intro":
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🧠 Brain Tumor Classification and Segmentation</h1>
            <p>Welcome to our AI-powered tool for detecting and segmenting brain tumors from MRI images.<br>Click below to get started.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("🚀 Start", key="start", use_container_width=True):
            st.session_state.page = "analyze"
            st.rerun()



elif st.session_state.page == "analyze":
    st.title("🧪 Analyze MRI Images")

    with st.sidebar:
        st.header("📋 Patient Info")
        name = st.text_input("Name", max_chars=50)
        age = st.number_input("Age", min_value=1, max_value=149, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        address = st.text_area("Address")
        submit_info = st.button("Save Patient Info")

        if submit_info:
            if name and address:
                st.session_state.patient_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "address": address
                }
                st.success("✅ Patient info saved!")
            else:
                st.warning("Please complete all fields.")

    uploaded_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    enable_gradcam = st.checkbox("Show Grad-CAM Explanation")

    if uploaded_files:
        if st.button("🔍 Analyze Images"):
            st.session_state.results = []
            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                img_bytes = file.getvalue()

                response = requests.post("http://localhost:8000/predict/", files={"file": img_bytes})
                if response.status_code == 200:
                    result = response.json()
                    overlay_bytes = bytes.fromhex(result["overlay_image"])
                    overlay_img = Image.open(io.BytesIO(overlay_bytes))
                    gradcam_img = None

                    if enable_gradcam:
                        cam_response = requests.post("http://localhost:8000/gradcam/", files={"file": img_bytes})
                        if cam_response.status_code == 200:
                            cam_data = cam_response.json()
                            cam_bytes = bytes.fromhex(cam_data["gradcam_image"])
                            gradcam_img = Image.open(io.BytesIO(cam_bytes))

                    st.session_state.results.append({
                        "filename": file.name,
                        "original": image,
                        "overlay": overlay_img,
                        "gradcam": gradcam_img,
                        "label": result['class_label'],
                        "confidence": result['confidence']
                    })

    if st.session_state.results:
        st.subheader("🖼️ Results")
        cols = st.columns(4)
        for idx, res in enumerate(st.session_state.results):
            col = cols[idx % 4]
            with col:
                img_to_show = res['gradcam'] if enable_gradcam and res['gradcam'] else res['overlay']
                st.image(img_to_show, caption=f"{res['label']} ({res['confidence']*100:.2f}%)", use_container_width=True)

        with st.sidebar:
            st.markdown("---")
            if st.button("⬇️ Download All Results (ZIP)"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    for res in st.session_state.results:
                        img_io = io.BytesIO()
                        res['overlay'].save(img_io, format='PNG')
                        zipf.writestr(f"{res['filename']}_overlay.png", img_io.getvalue())
                        if res['gradcam']:
                            img_io = io.BytesIO()
                            res['gradcam'].save(img_io, format='PNG')
                            zipf.writestr(f"{res['filename']}_gradcam.png", img_io.getvalue())
                st.download_button("📦 Download ZIP", data=zip_buffer.getvalue(), file_name="results.zip")

            if st.button("📝 Export Report (PDF)"):
                pdf = FPDF()
                pdf.add_font("DejaVu", "", "assets/font/DejaVuSans.ttf", uni=True)
                pdf.add_font("DejaVu", "B", "assets/font/DejaVuSans-Bold.ttf", uni=True)
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()


                pdf.set_font("DejaVu", "B", 16)
                pdf.cell(0, 10, "Brain Tumor Diagnosis Report", ln=True, align="C")
                pdf.ln(10)

                pdf.set_font("DejaVu", size=12)
                patient = st.session_state.patient_info
                pdf.cell(0, 10, f"Name: {patient.get('name', '')}", ln=True)
                pdf.cell(0, 10, f"Age: {patient.get('age', '')}", ln=True)
                pdf.cell(0, 10, f"Gender: {patient.get('gender', '')}", ln=True)
                pdf.multi_cell(0, 10, f"Address: {patient.get('address', '')}")
                pdf.ln(5)

                for idx, res in enumerate(st.session_state.results):
                    pdf.set_font("DejaVu", "B", 12)
                    pdf.cell(0, 10, f"Overlay Image (with Mask): {res['filename']}", ln=True)
                    pdf.set_font("DejaVu", size=12)
                    pdf.cell(0, 10, f"Prediction: {res['label']} ({res['confidence']*100:.2f}%)", ln=True)
                    pdf.ln(2)

                    img_io = io.BytesIO()
                    res['overlay'].save(img_io, format='PNG')
                    img_path = f"temp_overlay_{idx}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_io.getvalue())
                    pdf.image(img_path, w=120)
                    pdf.ln(10)
                    os.remove(img_path)

                pdf.output("temp_report.pdf")
                with open("temp_report.pdf", "rb") as f:
                    pdf_data = f.read()
                os.remove("temp_report.pdf")
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name="patient_report.pdf",
                    mime="application/pdf"
                )
