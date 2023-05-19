#import library ที่ต้องใช้ทั้งหมด
from fastai.vision.all import (
    load_learner,
    PILImage,
)
import glob
from random import shuffle
import urllib.request
from PIL import Image
import os

#import streamlit มาในชื่อ st เพื่อใช้ในการสร้าง user interface
import streamlit as st

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


MODEL_URL = "https://github.com/PakinDioxide/Dog-Breed-Classification/raw/main/models/dbc_resnet50_new_fastai.pkl"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner("model.pkl", cpu=True)

# เราจะแบ่งหน้าจอเป็น 
# 1. sidebar ประกอบด้วยตัวเลือกรูปภาพ
# 2. main page ประกอบด้วยรูปและคำทำนาย

##################################
# main page
##################################

# ใส่ title ของ main page
st.title("Dog Breed Classification")


##################################
# sidebar
##################################

#function การทำนาย
def predict(img, learn):
    # ย่อขนาดรูป
    pimg = img.resize([224,224])
    
    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(pimg)
    
    pred = ' '.join(pred.split('_')[1:])

    # โชว์ผลการทำนาย
    st.success(f'This is "{pred} Dog" with the probability of {pred_prob[pred_idx]*100:.02f}%')
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)

# ใส่ title ของ sidebar
st.sidebar.write('# Upload a dog image to classify!')

# radio button สำหรับเลือกว่าจะทำนายรูปจาก validation set หรือ upload รูปเอง
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])
# โหลดรูปจาก validation set แล้ว shuffle
valid_images = glob.glob('images/valid/*/*')
shuffle(valid_images)

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)
    
    if fname is None:
        st.sidebar.write("Please select an image...")
    else:
        # เปิดรูป
        img = Image.open(fname)
        img = im.convert('RGB')
        img.save('fname.jpg')
        
        img = Image.open('fname.jpg')
        
        st.sidebar.image(img, 'Is this the image you want to predict?')
        st.sidebar.write(img.format)

        if st.sidebar.button("Predict Now!"):
            # เรียก function ทำนาย
            predict(img, learn_inf)

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['jpg', 'jpeg', 'png'],
                                     accept_multiple_files=False)
    if fname is None:
        st.sidebar.write("Please select an image...")
    else:
        # เปิดรูป
        img = Image.open(fname)
        
        st.sidebar.image(img, 'Is this the image you want to predict?')

        if st.sidebar.button("Predict Now!"):
            # เรียก function ทำนาย
            predict(img, learn_inf)
