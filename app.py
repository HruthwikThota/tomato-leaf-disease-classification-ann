import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Tomato Leaf Disease Classifier (ANN)",
    page_icon="🍅",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato_ann_model.keras")

model = load_model()

CLASS_NAMES = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites Two-spotted Spider Mite',
    'Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Healthy'
]

st.title("🍅 Tomato Leaf Disease Classification (ANN)")
st.write("Upload an image of a tomato leaf to predict whether it is healthy or diseased using an Artificial Neural Network.")

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Updated to (64, 64) to match training IMG_SIZE
    image_resized = image.resize((64, 64))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    
    predicted_disease = CLASS_NAMES[predicted_class_idx]
    
    st.success(f"**Prediction:** {predicted_disease}")
    st.info(f"**Confidence:** {confidence:.2f}%")
