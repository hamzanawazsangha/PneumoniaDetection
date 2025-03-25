# Pneumonia Detection from Chest X-rays 🩺

![Pneumonia Detection Demo](demo.gif) *(Example GIF showing the app in action)*

## Overview
This project is a deep learning-based system that detects pneumonia from chest X-ray images using a CNN model with EfficientNetV2B1 architecture. The system provides instant predictions with confidence scores, helping medical professionals in preliminary diagnosis.

## Key Features
- 🖼️ **Image Upload**: Accepts chest X-ray images in JPG, JPEG, or PNG formats
- ⚡ **Instant Prediction**: Provides results in seconds
- 📊 **Confidence Score**: Shows prediction certainty percentage
- 🎨 **Visual Feedback**: Color-coded results (red for pneumonia, green for normal)
- 📱 **Responsive Design**: Works on both desktop and mobile devices

## Technical Details
- **Model Architecture**: EfficientNetV2B1 with transfer learning
- **Input Size**: 256x256 pixels RGB images
- **Output**: Binary classification (Normal/Pneumonia)
- **Framework**: TensorFlow 2.x
- **Frontend**: Streamlit

## How to Use
1. **Run the app locally**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
2. **Access the web interface**:
The app will open in your default browser at http://localhost:8501
Click "Browse files" to upload a chest X-ray image
View the prediction results instantly

3. **Interpret results**:

🟢 **Normal**: No signs of pneumonia detected

🔴 **Pneumonia Detected**: Patterns consistent with pneumonia identified

Confidence percentage indicates model certainty

## File Structure
```
pneumonia-detection/
├── app.py                # Main application code
├── pneumoniamodel.h5     # Trained model weights
├── requirements.txt      # Python dependencies
└── README.md             # This documentation
```
### Requirements
- **Python 3.8+**
- **TensorFlow 2.x**
- **Streamlit**
- **Pillow**
- **NumPy**

### Install all dependencies:

```bash
pip install -r requirements.txt
```
### Limitations
- This is an AI-assisted tool, not a replacement for professional medical diagnosis
- Performance depends on image quality and positioning
- Model trained on specific datasets - may not generalize to all populations

### Future Improvements
- Add multi-class classification for different pneumonia types
- Implement DICOM support for direct hospital system integration
- Add patient history tracking
- Develop mobile app version

### Acknowledgments
- **Dataset**: Chest X-Ray Images (Pneumonia) from Kaggle
- Model architecture based on **EfficientNetV2**
