# ===========================================
# FILE: bp_report_ui.py
# ===========================================
# AI Blood Pressure Analyzer — Real Life Implementation
# Beautiful Streamlit UI
# ===========================================

import streamlit as st
from PIL import Image
import pytesseract
import re
import numpy as np
import joblib

# ---------------------------
# Helper: Categorize BP
# ---------------------------
def bp_category(s, d):
    if s > 180 or d > 120:
        return "Hypertensive Crisis ⚠️"
    elif s >= 140 or d >= 90:
        return "Hypertension Stage 2 🟥"
    elif 130 <= s < 140 or 80 <= d < 90:
        return "Hypertension Stage 1 🟧"
    elif 120 <= s < 130 and d < 80:
        return "Elevated 🟨"
    elif s < 120 and d < 80:
        return "Normal 🟩"
    return "Uncategorized"

# ---------------------------
# Extract BP values from text
# ---------------------------
def extract_bp_from_text(text):
    # Pattern like "BP 140/90" or "Systolic: 140 Diastolic: 90"
    m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Named fields fallback
    s = re.search(r'(systolic|SBP)[^\d]{0,10}(\d{2,3})', text, re.I)
    d = re.search(r'(diastolic|DBP)[^\d]{0,10}(\d{2,3})', text, re.I)
    if s and d:
        return int(s.group(2)), int(d.group(2))
    return None, None

# ---------------------------
# UI Layout
# ---------------------------
st.set_page_config(page_title="AI Blood Pressure Report Analyzer", layout="wide")
st.title("💉 AI Blood Pressure Report Analyzer")
st.markdown("#### Upload a **report image or text**, and let AI detect your blood pressure levels.")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_file = st.sidebar.file_uploader("Upload Trained Model (.joblib)", type=["joblib"])

st.sidebar.markdown("---")
st.sidebar.info("Model trained using real dataset.\n\nUse `train_bp_model.py` to train and save your model first.")

# Main options
option = st.radio("Choose Input Type", ["🖼️ Image Report", "📝 Text Report", "⌨️ Manual Input"])

if model_file:
    model_data = joblib.load(model_file)
    model = model_data["model"]
    features = model_data["features"]
else:
    model = None
    st.warning("Upload the trained model to enable predictions.")

# ---------------------------
# Image Report (OCR)
# ---------------------------
if option == "🖼️ Image Report":
    uploaded_image = st.file_uploader("Upload Report Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Report", use_container_width=True)

        with st.spinner("🔍 Extracting text using OCR..."):
            extracted_text = pytesseract.image_to_string(img)
            systolic, diastolic = extract_bp_from_text(extracted_text)

        if systolic and diastolic:
            st.success(f"✅ Detected BP: **{systolic}/{diastolic} mmHg**")
            category = bp_category(systolic, diastolic)
            st.markdown(f"### 🩺 Category: **{category}**")

            if model:
                X = np.array([[systolic, diastolic]])
                prediction = model.predict(X)[0]
                st.info(f"Model-based classification: **{prediction}**")

        else:
            st.error("❌ Could not detect BP values. Try a clearer image.")

# ---------------------------
# Text Report
# ---------------------------
elif option == "📝 Text Report":
    text_input = st.text_area("Paste report text here:")
    if st.button("Analyze Text"):
        systolic, diastolic = extract_bp_from_text(text_input)
        if systolic and diastolic:
            st.success(f"✅ Extracted BP: {systolic}/{diastolic} mmHg")
            category = bp_category(systolic, diastolic)
            st.markdown(f"### 🩺 Category: **{category}**")

            if model:
                X = np.array([[systolic, diastolic]])
                prediction = model.predict(X)[0]
                st.info(f"Model-based classification: **{prediction}**")

        else:
            st.warning("Could not find BP numbers in the text.")

# ---------------------------
# Manual Input
# ---------------------------
elif option == "⌨️ Manual Input":
    s = st.number_input("Enter Systolic (mmHg)", min_value=60, max_value=250, value=120)
    d = st.number_input("Enter Diastolic (mmHg)", min_value=40, max_value=150, value=80)
    if st.button("Classify"):
        category = bp_category(s, d)
        st.markdown(f"### 🩺 Category: **{category}**")
        if model:
            X = np.array([[s, d]])
            prediction = model.predict(X)[0]
            st.info(f"Model-based classification: **{prediction}**")
