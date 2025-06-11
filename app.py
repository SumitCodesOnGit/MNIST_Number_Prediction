# Streamlit UI for Number Prediction

# import necessary packages
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# load model
model = tf.keras.models.load_model("E:/workspace_neural_network/MNIST_Number_Prediction/mnist_model.h5")

# Give title and heading for UI
st.title("MNIST Digit Classifier")
st.markdown("Upload a 28*28 pixel image of a digit (0-9) to predict.")

# Upload image u want to classify
uploaded_file = st.file_uploader("Choose an image...", type=['png','jpg','jpeg'])

# preprocessing function for input image
def preprocess_image(image):
    image = image.convert('L')   # to grayscale
    image = image.resize((28,28))  # to 28*28 pixel
    img_array = np.array(image) / 255.0  # normalize
    return img_array

# Make and display prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=False)

    preprocessed_image = preprocess_image(image)
    prediction_input = preprocessed_image.reshape(1,28,28)

    prediction = model.predict(prediction_input)
    predicted_digit = np.argmax(prediction)

    st.markdown(f" Predicted Digit: {predicted_digit}")





