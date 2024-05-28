import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('my__model.keras')

# Define the image size and class names
image_size = (128, 128)
class_names = ['fake', 'real']

def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def predict_image(image_path):
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    return class_names[y_pred_class], np.amax(y_pred) * 100

# Streamlit UI
st.title("Image Real vs Fake Classifier")

image_path = st.text_input("Enter the image path:")
if st.button("Classify Image"):
    if os.path.exists(image_path):
        prediction, confidence = predict_image(image_path)
        st.write(f'Class: {prediction}, Confidence: {confidence:.2f}%')
        st.image(image_path, caption=f'Class: {prediction}, Confidence: {confidence:.2f}%')
    else:
        st.write("The specified image path does not exist.")
