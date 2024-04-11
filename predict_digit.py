import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import zipfile
from PIL import Image, ImageOps, ImageEnhance
import streamlit as st

st.title("Digit Classification")
#st.write("Upload an image of a handwritten digit.  The image should be .jpg, .jpeg, or .png")

def preprocess_image(image_path, brightness=1, contrast=1):
    image = Image.open(image_path)
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness) 

    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)  

    image = image.convert("L")
    image = ImageOps.invert(image)

    image = image.resize((28,28))

    # convert to numpy array
    image_array = np.asarray(image)
    image_array = image_array / 255.0

    image_array = image_array.reshape(1,784)

    return image_array

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with st.expander('View image'):
    
    st.write('Choose an enhancement factor greater than 1 to increase the brightness or contrast')
    brightness = st.number_input('Enter the brightness factor',min_value=0.0, value=2.0, step=0.1)
    contrast = st.number_input('Enter the contrast factor',min_value=0.0, value=2.0, step=0.1)

    if uploaded_file is not None:
        X_new = preprocess_image(uploaded_file, brightness=brightness, contrast = contrast)
        fig, ax = plt.subplots(figsize=(1,1))
        ax.imshow(X_new.reshape(28,28), cmap='binary', interpolation='nearest')
        ax.axis('off')
        st.pyplot(fig)

with zipfile.ZipFile('my_model.zip', 'r') as zip_ref:
    zip_ref.extractall()
model = load('my_model.joblib')

run_model = st.button('Predict digit')

if run_model:
    prob = model.predict_proba(X_new)[0]
    top_two_indices = np.argsort(prob)[::-1][:2]  

    top_two_probs = prob[top_two_indices]


    top_two_classes = top_two_indices  

    # Print or return the result

    st.write(f"### Predicted class: {top_two_classes[0]} (probability = {top_two_probs[0]})")
    st.write(f"Next highest predicted class: {top_two_classes[1]} (probability = {top_two_probs[1]})")
