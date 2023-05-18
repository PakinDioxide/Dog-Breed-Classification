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

# โหลดโมเดลจากแหล่งข้อมูลในอินเตอร์เน็ตเพื่อประหยัดพื้นที่เวลา deploy บน heroku
# MODEL_URL = "https://github.com/PakinDioxide/Dog-Breed-Classification/raw/main/models/dbc_resnet50_new_fastai.pkl"
# urllib.request.urlretrieve(MODEL_URL, "model.pkl")
# learn_inf = load_learner('model.pkl', cpu=True)

# ใส่ title ของ sidebar
st.sidebar.write('### Enter a dog image to classify')

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
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)
    if not fname == None:
        img = Image.open(fname).resize([224,224])

##################################
# main page
##################################

# ใส่ title ของ main page
st.title("Chocolate Chip vs Raisin Cookies")
st.write(os.listdir('/app/dog-breed-classification/'))

#function การทำนาย
def predict(img, learn):

    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(img)

    # โชว์ผลการทำนาย
    st.success(f"This is {pred} with the probability of {pred_prob[pred_idx]*100:.02f}%")
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)

# เปิดรูป
img = PILImage.create(fname)

# เรียก function ทำนาย
predict(img, learn_inf)
