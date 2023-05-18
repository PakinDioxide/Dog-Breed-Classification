import streamlit as st
from PIL import Image
from fastai.vision.all import *
import torch

# Load the model
MODEL_URL = 'https://github.com/PakinDioxide/Dog-Breed-Classification/raw/main/models/dbc_resnet50_new_fastai.pkl'
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)
