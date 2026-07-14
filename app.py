# app.py
import streamlit as st

# ============================
# PAGE CONFIG - MUST BE FIRST
# ============================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# ============================
# IMPORTS (after config)
# ============================
import cv2
import numpy as np
from skimage.feature import hog
import joblib
import os

# ============================
# LOAD SAVED ARTIFACTS
# ============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le

# Try to load model; show error if files are missing
try:
    model, scaler, le = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model files not found. Please run `train_model.py` first to generate the required `.pkl` files.")

# ============================
# CONFIGURATION
# ============================
IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ============================
# PREPROCESSING FUNCTION
# ============================
def preprocess_image(img):
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Resize
    resized = cv2.resize(gray, IMG_SIZE)
    # HOG features
    hog_feat = hog(resized, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   transform_sqrt=True, block_norm='L2-Hys')
    return hog_feat

# ============================
# STREAMLIT UI
# ============================
st.title("🧠 Brain Tumor Classification using MRI")
st.markdown("Upload an MRI scan image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display image – FIXED parameter name
    st.image(img, caption="Uploaded MRI", use_container_width=True)   # <-- changed from use_container_width
    
    with st.spinner("Analyzing..."):
        # Preprocess
        features = preprocess_image(img)
        features_scaled = scaler.transform([features])
        
        # Predict
        pred_label_enc = model.predict(features_scaled)[0]
        pred_class = le.inverse_transform([pred_label_enc])[0]
        
        # Get probabilities (if supported)
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba) * 100
        
        # Show result
        st.success(f"**Prediction:** {pred_class}")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show probability breakdown
        st.write("**Class Probabilities:**")
        for cls, prob in zip(le.classes_, proba):
            st.write(f"- {cls}: {prob*100:.2f}%")
elif uploaded_file is not None and not model_loaded:
    st.warning("Cannot make predictions because model files are missing.")
else:
    st.info("Please upload an MRI image to get a prediction.")