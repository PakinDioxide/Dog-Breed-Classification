import streamlit as st
import pandas as pd
from PIL import Image

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = Image.open(uploaded_file)

img.show()
