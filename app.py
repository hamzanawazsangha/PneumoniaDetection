import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown

# Model loading with error handling
model_url = "https://drive.google.com/uc?id=165LXJNpFIyDKqK6ZJJ8j_XDuuDMKr2kp"
model_path = "pneumoniamodel.h5"

@st.cache_resource
def load_pneumonia_model():
    try:
        # Download model if not exists
        if not os.path.exists(model_path):
            gdown.download(model_url, model_path, quiet=False)
        
        # Try normal loading first
        try:
            model = load_model(model_path)
        except:
            # Fallback to loading without compilation
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_pneumonia_model()

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #2a9d8f;
        text-align: center;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    .pneumonia {
        background-color: #ffecec;
        border-left: 5px solid #e76f51;
    }
    .normal {
        background-color: #e8f5e9;
        border-left: 5px solid #2a9d8f;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.markdown("### Upload a chest X-ray image to check for pneumonia")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    try:
        prediction = model.predict(img_array)
        is_pneumonia = prediction[0][0] > 0.5
        result = "Pneumonia Detected! 🚨" if is_pneumonia else "Normal Chest X-ray ✅"
        confidence = prediction[0][0] if is_pneumonia else 1 - prediction[0][0]
        
        # Display result with emojis and styled box
        box_class = "pneumonia" if is_pneumonia else "normal"
        st.markdown(
            f"""<div class="prediction-box {box_class}">
                <h2 style='color: {'#e76f51' if is_pneumonia else '#2a9d8f'};'>{result}</h2>
                <p><b>Confidence:</b> {confidence*100:.2f}%</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Additional information
        with st.expander("ℹ️ What do these results mean?"):
            st.markdown("""
            - **Normal Chest X-ray**: No signs of pneumonia detected
            - **Pneumonia Detected**: The model has identified patterns consistent with pneumonia
            - **Confidence score**: Shows how certain the model is about the prediction
            """)
            
            st.markdown("""
            **Note:** This is an AI-assisted diagnosis tool. Always consult with a healthcare professional 
            for medical diagnosis and treatment.
            """)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
