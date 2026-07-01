
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title='Road Accident Severity Classifier', layout='centered')

st.title('Road Accident Severity Classification')
st.write('Upload a road accident image. The model predicts whether the severity is Minor Impact, Substantial Impact, or Critical Impact.')

MODEL_PATH = 'best_accident_severity_model.keras'
CLASS_NAMES = ['Minor Impact', 'Substantial Impact', 'Critical Impact']
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader('Upload accident image', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    resized = image.resize(IMG_SIZE)
    arr = np.array(resized).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(probs))

    st.subheader('Prediction Result')
    st.success(f'Predicted Severity: {pred_class}')
    st.write(f'Confidence: {confidence:.2%}')

    st.subheader('Class Probabilities')
    for cls, prob in zip(CLASS_NAMES, probs):
        st.write(f'{cls}: {prob:.2%}')
