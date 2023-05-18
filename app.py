import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from fastai.vision.all import *

# Load the model
MODEL_URL = 'https://github.com/cstorm125/choco-raisin/raw/main/notebooks/models/resnet34_finetune1e3_5p.pkl'
model_file = 'dbc_resnet50_new_fastai.pkl'
if not Path(model_file).exists():
    r = requests.get(MODEL_URL)
    with open(model_file, 'wb') as f:
        f.write(r.content)

learn_inf = load_learner(model_file, cpu=True)

# Page title
st.title("Dog Breed Classification")

# Image upload and prediction
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction
    pred, pred_idx, pred_prob = learn_inf.predict(image)
    breed = pred.capitalize()
    probability = pred_prob[pred_idx] * 100

    # Display the predicted dog breed and probability
    st.success(f"Predicted Breed: {breed}")
    st.info(f"Probability: {probability:.2f}%")
