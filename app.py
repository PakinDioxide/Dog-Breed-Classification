import streamlit as st
import sys
import os
import glob
from PIL import Image

# --- 1. CRITICAL FIX FOR 2023 MODELS (The Bridge) ---
# This MUST happen before 'from fastai...' to prevent ImportErrors
try:
    import fastcore.transform
    import fasttransform
    if not hasattr(fastcore.transform, 'Pipeline'):
        fastcore.transform.Pipeline = fasttransform.Pipeline
        fastcore.transform.Transform = fasttransform.Transform
except ImportError:
    st.error("Missing dependencies. Please ensure 'fasttransform' and 'ipython' are in requirements.txt")

# --- 2. NOW IMPORT FASTAI ---
from fastai.vision.all import *

# --- 3. MODEL LOADING WITH CACHING ---
@st.cache_resource
def get_model():
    # Streamlit Cloud default path
    model_path = '/mount/src/dog-breed-classification/models/dbc_resnet50_new_fastai.pkl'
    # Local development fallback
    if not os.path.exists(model_path):
        model_path = 'models/dbc_resnet50_new_fastai.pkl'
    
    return load_learner(model_path, cpu=True)

learn_inf = get_model()

# --- 4. PREDICTION LOGIC ---
def predict(img, learn):
    # Ensure image is RGB
    img = img.convert("RGB")
    pimg = PILImage.create(img)

    pred, pred_idx, pred_prob = learn.predict(pimg)

    # Label processing
    pred_name = pred.split('_')[1:]
    if pred_name[-1] == 'Dog':
        display_name = ' '.join(pred_name[:-1])
    else:
        display_name = ' '.join(pred_name)

    st.success(
        f'This is "{display_name} Dog" with the probability of {pred_prob[pred_idx]*100:.02f}%'
    )
    st.image(img, use_container_width=True)
    st.balloons()

# ใส่ title ของ main page
st.title("Dog Breed Classification")

# ใส่ title ของ sidebar
st.sidebar.write('# Upload a dog image to classify!')

# radio button สำหรับเลือกว่าจะทำนายรูปจาก validation set หรือ upload รูปเอง
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image', 'Take a photo'])

# Rotation
if 'rotation' not in st.session_state:
    st.session_state.rotation = 0

# โหลดรูปจาก validation set แล้ว shuffle
valid_images = glob.glob('/mount/src/dog-breed-classification/images/test/*/*')
valid_images.sort()
for i in range(len(valid_images)):
    k = str(valid_images[i])
    k =k.replace('/mount/src/dog-breed-classification/images/test/', '')
    valid_images[i] = k

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)
    
    # เปิดรูป
    img = Image.open(f'/mount/src/dog-breed-classification/images/test/{fname}')
    img = img.rotate(st.session_state.rotation, expand=True)

    st.sidebar.image(img, f'Is this the image you want to predict?', use_container_width=True)

    if st.sidebar.button("Rotate Image"):
        st.session_state.rotation = (st.session_state.rotation - 90) % 360
        st.rerun()
    
    if st.sidebar.button("Predict Now!"):
        # เรียก function ทำนาย
        predict(img, learn_inf)
        
elif option == 'Use your own image':
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['jpg', 'jpeg', 'png'],
                                     accept_multiple_files=False)
    if fname is None:
        st.sidebar.write("Please select an image...")
    else:
        # เปิดรูป
        img = Image.open(fname)
        # เปลี่ยน format ภาพ
        img = img.convert('RGB')
        img.save('fname.jpg')
        
        img = Image.open('fname.jpg')
        img = img.rotate(st.session_state.rotation, expand=True)
        
        st.sidebar.image(img, f'Is this the image you want to predict?', use_container_width=True)

        if st.sidebar.button("Rotate Image"):
            st.session_state.rotation = (st.session_state.rotation - 90) % 360
            st.rerun()
        
        if st.sidebar.button("Predict Now!"):
            # เรียก function ทำนาย
            predict(img, learn_inf)
else:
        fname = st.sidebar.camera_input('Take a photo of a dog')
        if fname is None:
            st.sidebar.write("Please take a photo...")
        else:
            # เปิดรูป
            img = Image.open(fname)
            # เปลี่ยน format ภาพ
            img = img.convert('RGB')
            img.save('fname.jpg')

            img = Image.open('fname.jpg')
            img = img.rotate(st.session_state.rotation, expand=True)

            st.sidebar.image(img, 'Is this the image you want to predict?', use_container_width=True)

            if st.sidebar.button("Rotate Image"):
                st.session_state.rotation = (st.session_state.rotation - 90) % 360
                st.rerun()
            
            if st.sidebar.button("Predict Now!"):
                # เรียก function ทำนาย
                predict(img, learn_inf)


