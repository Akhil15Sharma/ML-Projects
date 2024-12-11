import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

# Parameters
image_size = 224
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Load the model
model_path = 'effnet.keras'
model = tf.keras.models.load_model(model_path)

st.title('Brain Tumor Classification App')
st.write("Upload an MRI image to classify it as one of the following categories:")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and preprocess the image
    img = Image.open(uploaded_file)
    img = img.convert('RGB')  # Ensure image is in RGB mode
    img = img.resize((image_size, image_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    pred = model.predict(img_array)
    pred_probs = tf.nn.softmax(pred[0]).numpy()
    pred_class = np.argmax(pred, axis=1)[0]
    
    # Display results
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {categories[pred_class]}")
    
    # Display prediction probabilities
    st.write("Prediction probabilities:")
    for i, category in enumerate(categories):
        st.write(f"{category}: {pred_probs[i]*100:.2f}%")
