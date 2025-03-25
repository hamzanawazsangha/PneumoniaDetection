import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the saved model
import gdown  # for Google Drive

# Add this at the start of your app
model_url = "https://drive.google.com/file/d/165LXJNpFIyDKqK6ZJJ8j_XDuuDMKr2kp/view?usp=sharing"
model_path = "pneumoniamodel.h5"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, fuzzy=True)

model = tf.keras.models.load_model(model_path)

# Load evaluation data
with open('roc_data_binary_class.pkl', 'rb') as f:
    roc_data = pickle.load(f)
    
with open('conf_matrix.pkl', 'rb') as f:
    conf_matrix = pickle.load(f)
    
with open('report_df.pkl', 'rb') as f:
    report_df = pickle.load(f)

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
</style>
""", unsafe_allow_html=True)

# Main app
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.markdown("### Upload a chest X-ray image to check for pneumonia")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    result = "Pneumonia Detected! 🚨" if prediction[0][0] > 0.5 else "Normal Chest X-ray ✅"
    
    # Display result with emojis
    st.markdown(f"<h2 style='text-align: center; color: {'#e76f51' if prediction[0][0] > 0.5 else '#2a9d8f'};'>{result}</h2>", 
                unsafe_allow_html=True)
    
    # Confidence meter
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
    
    # Additional information
    with st.expander("ℹ️ What do these results mean?"):
        st.markdown("""
        - **Normal Chest X-ray**: No signs of pneumonia detected
        - **Pneumonia Detected**: The model has identified patterns consistent with pneumonia
        - Confidence score shows how certain the model is about the prediction
        """)

# Developer section
st.markdown('<div class="developer-section" onclick="document.getElementById(\'dev-results\').style.display=\'block\'">👨💻 Developer Results</div>', unsafe_allow_html=True)

# Hidden developer results (initially hidden)
st.markdown('<div id="dev-results" style="display:none">', unsafe_allow_html=True)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, thresholds = roc_data
plt.figure()
plt.plot(fpr, tpr, color='#2a9d8f', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
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

# Add custom JavaScript to handle the developer section toggle
st.markdown("""
<script>
    document.querySelector('.developer-section').addEventListener('click', function() {
        var devResults = document.getElementById('dev-results');
        if (devResults.style.display === 'none') {
            devResults.style.display = 'block';
        } else {
            devResults.style.display = 'none';
        }
    });
</script>
""", unsafe_allow_html=True)
