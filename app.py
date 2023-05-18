import streamlit as st
from PIL import Image
from fastai.vision.all import *
import pickle

# Load the model
MODEL_URL = 'https://github.com/PakinDioxide/Dog-Breed-Classification/raw/main/models/dbc_resnet50_new_fastai.pkl'
path = Path('models')
path.mkdir(parents=True, exist_ok=True)
model_file = path/'dbc_resnet50_new_fastai.pkl'

if not model_file.exists():
    download_url(MODEL_URL, model_file)

# Load the model using pickle
learn_inf = pickle.load(open(model_file, 'rb'))

# Sidebar
st.sidebar.title('Enter a dog to classify')
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])

if option == 'Use a validation image':
    st.sidebar.title('Select a validation image')
    valid_images = get_image_files('images/valid')
    fname = st.sidebar.selectbox('', valid_images)
else:
    st.sidebar.title('Select an image to upload')
    uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
    else:
        valid_images = get_image_files('images/valid')
        img = PILImage.create(valid_images[0])

# Main page
st.title('Dog Breed Classification')

def predict(img):
    pred, pred_idx, pred_prob = learn_inf.predict(img)
    st.success(f"This is {pred} with a probability of {pred_prob[pred_idx]*100:.02f}%")
    st.image(img.to_thumb(300, 300))

predict(img)
