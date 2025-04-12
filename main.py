import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Define model paths
MODEL_PATHS = {
    "Brain": "models/Brain_vgg16_model.h5",
    "Breast": "models/Breast_vgg16_model.h5",
    "Kidney": "models/Kidney_vgg16_model.h5",
    "Liver": "models/Liver_vgg16_model.h5",
    "Lung": "models/Lung_vgg16_model.h5"
}

# Define class labels for each organ
organ_classes = {
    "Brain": ["Glioma", "Meningioma", "No Tumor","Pituitary Tumor"],
    "Breast": [ "Benign","Malignant", "Normal"],
    "Kidney": ["Cyst","Normal", "Stone", "Tumor" ],
    "Liver": ["Cirrhosis", "No Fibrosis", "Periportal Fibrosis", "Portal Fibrosis", "Septal Fibrosis"],
    "Lung": ["Lung_Opacity", "normal", "Viral Pneumonia"]
}

# Streamlit UI
import streamlit as st

# Custom CSS for styling
import streamlit as st

# Custom CSS for styling

# Display the styled title and subtitle
st.title("Multi-Organ Disease Detection")
st.write("Upload a radiology image and select the organ for disease classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Organ selection
target_organ = st.selectbox("Select the organ:", list(MODEL_PATHS.keys()))

if uploaded_file and target_organ:
    # Load the selected model
    model = load_model(MODEL_PATHS[target_organ])

    # Process the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_label = organ_classes[target_organ][predicted_class]

    # Display image & result
    st.image(uploaded_file, caption=f"Uploaded {target_organ} Image", use_column_width=True)
    st.write(f"**Predicted Disease:** {class_label}")
