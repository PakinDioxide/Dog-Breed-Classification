import streamlit as st
import pandas as pd
from PIL import Image
import urllib
from fastbook import *


uploaded_file = st.file_uploader("Choose a dog image to predict!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)
    
MODEL_URL = 'https://github.com/PakinDioxide/Dog-Breed-Classification/blob/main/models/dbc_resnet50_new_fastai.pkl'
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)
