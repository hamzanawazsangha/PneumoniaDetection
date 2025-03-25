import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

# Load evaluation data with error handling
@st.cache_data
def load_evaluation_data():
    try:
        with open('roc_data_binary_class.pkl', 'rb') as f:
            roc_data = pickle.load(f)
        
        with open('conf_matrix.pkl', 'rb') as f:
            conf_matrix = pickle.load(f)
        
        with open('report_df.pkl', 'rb') as f:
            report_df = pickle.load(f)
            
        return roc_data, conf_matrix, report_df
    except Exception as e:
        st.error(f"Error loading evaluation data: {str(e)}")
        return None, None, None

roc_data, conf_matrix, report_df = load_evaluation_data()

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
    .developer-section {
        color: #666;
        font-size: 0.9em;
        text-align: center;
        margin-top: 50px;
        cursor: pointer;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    #dev-results {
        display: none;
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

# Developer section
try:
    if roc_data is not None and conf_matrix is not None and report_df is not None:
        st.markdown(
            '<div class="developer-section" onclick="toggleDevResults()">'
            '👨💻 Developer Results</div>', 
            unsafe_allow_html=True
        )

        # Hidden developer results (initially hidden via CSS)
        st.markdown('<div id="dev-results">', unsafe_allow_html=True)
        
        # ROC Curve with your saved format
        st.subheader("ROC Curve")
        plt.figure(figsize=(10, 8))
        plt.plot(roc_data["fpr"], roc_data["tpr"], 
                label=f'ROC Curve (AUC = {roc_data["roc_auc"]:.2f})', 
                color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                   yticklabels=['Actual Normal', 'Actual Pneumonia'])
        plt.title('Confusion Matrix')
        st.pyplot(plt)

        # Classification Report
        st.subheader("Classification Report")
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))

        st.markdown('</div>', unsafe_allow_html=True)

        # JavaScript to handle the toggle
        st.markdown("""
        <script>
        function toggleDevResults() {
            var devResults = document.getElementById('dev-results');
            if (devResults.style.display === 'none') {
                devResults.style.display = 'block';
            } else {
                devResults.style.display = 'none';
            }
        }
        </script>
        """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error displaying developer results: {str(e)}")
