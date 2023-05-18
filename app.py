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

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


MODEL_URL = "https://github.com/PakinDioxide/Dog-Breed-Classification/raw/main/models/dbc_resnet50_new_fastai.pkl"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)

# เราจะแบ่งหน้าจอเป็น 
# 1. sidebar ประกอบด้วยตัวเลือกรูปภาพ
# 2. main page ประกอบด้วยรูปและคำทำนาย

##################################
# sidebar
##################################

#function การทำนาย
def predict(img, learn):

    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(img)
    
    pred = ' '.join(pred.split('_')[1:])

    # โชว์ผลการทำนาย
    st.success(f'This is "{pred} Dog" with the probability of {pred_prob[pred_idx]*100:.02f}%')
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)

# ใส่ title ของ sidebar
st.sidebar.write('### Enter cookie to classify')

# radio button สำหรับเลือกว่าจะทำนายรูปจาก validation set หรือ upload รูปเอง
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])
# โหลดรูปจาก validation set แล้ว shuffle
valid_images = glob.glob('images/valid/*/*')
shuffle(valid_images)

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['jpg'],
                                     accept_multiple_files=False)
    if fname is None:
        fname = valid_images[0]
    else:
        # เปิดรูป
        img = Image.open(fname).resize([224, 224])

        # เรียก function ทำนาย
        predict(img, learn_inf)

##################################
# main page
##################################

# ใส่ title ของ main page
st.title("Dog Breed Classification")
