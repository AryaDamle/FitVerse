import streamlit as st
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF reading
import io
import re
import joblib
import pandas as pd

# Optional: Set Tesseract path (Windows users only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Load Trained Model ---
@st.cache_resource
def load_model():
    return joblib.load("bp_classifier.pkl")

model = load_model()

# --- Helper Functions ---
def extract_text_from_pdf(pdf_bytes):
    """Extracts text from PDF using PyMuPDF."""
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text("text")
    return text

def extract_text_from_image(image):
    """Extracts text from image using Tesseract OCR."""
    text = pytesseract.image_to_string(image)
    return text

def extract_bp_values(text):
    """Extract BP values (Systolic/Diastolic) using regex."""
    text = text.replace("\n", " ")
    match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text)
    if match:
        sy = int(match.group(1))
        di = int(match.group(2))
        return sy, di
    return None, None

def predict_bp_category(bmi, sy, di):
    """Predicts BP category using the trained model."""
    input_df = pd.DataFrame([[bmi, sy, di]], columns=["BMI", "BP_Sy", "BP_Di"])
    return model.predict(input_df)[0]

# --- Streamlit UI ---
st.set_page_config(page_title="AI Blood Pressure Analyzer 💓", page_icon="💉", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff5f6d, #ffc371);
            color: white;
            border-radius: 8px;
            height: 3em;
            font-size: 16px;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💉 AI Blood Pressure Report Analyzer")
st.markdown("### Upload your report (PDF or Image) to detect and analyze your blood pressure instantly!")

uploaded_file = st.file_uploader("📂 Upload BP Report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    with st.spinner("Extracting text from your report..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            text = extract_text_from_image(image)

    bp_sy, bp_di = extract_bp_values(text)
    
    if bp_sy and bp_di:
        st.success(f"✅ Detected BP: **{bp_sy}/{bp_di} mmHg**")

        bmi = st.number_input("Enter BMI (optional, default = 22)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)

        if st.button("🔍 Analyze Blood Pressure"):
            category = predict_bp_category(bmi, bp_sy, bp_di)
            if category == "Normal":
                st.success(f"🩺 Your Blood Pressure is **{category}** ✅\nStay healthy and maintain your lifestyle!")
            elif category == "High":
                st.error(f"⚠️ Your Blood Pressure is **{category}**!\nPlease consult a doctor soon.")
            else:
                st.warning(f"⬇️ Your Blood Pressure is **{category}**.\nEat well and stay hydrated.")
    else:
        st.error("❌ Could not detect BP values from your report.")
        st.markdown("You can also enter them manually below 👇")

        bp_sy = st.number_input("Systolic (BP_Sy)", min_value=60, max_value=200, value=120)
        bp_di = st.number_input("Diastolic (BP_Di)", min_value=40, max_value=130, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)

        if st.button("🔍 Analyze Manually Entered Values"):
            category = predict_bp_category(bmi, bp_sy, bp_di)
            if category == "Normal":
                st.success(f"🩺 Your Blood Pressure is **{category}** ✅")
            elif category == "High":
                st.error(f"⚠️ Your Blood Pressure is **{category}**!")
            else:
                st.warning(f"⬇️ Your Blood Pressure is **{category}**.")

st.markdown("---")
st.markdown("💡 *FitVerse: Blood Pressurer Reader*")
