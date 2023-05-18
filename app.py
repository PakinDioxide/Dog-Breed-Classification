#import library ที่ต้องใช้ทั้งหมด
from fastai.vision.all import (
    load_learner,
    PILImage,
)
import glob
from random import shuffle
import urllib.request

#import streamlit มาในชื่อ st เพื่อใช้ในการสร้าง user interface
import streamlit as st

# โหลดโมเดลจากแหล่งข้อมูลในอินเตอร์เน็ตเพื่อประหยัดพื้นที่เวลา deploy บน heroku
MODEL_URL = "https://github.com/cstorm125/choco-raisin/raw/main/notebooks/models/resnet34_finetune1e3_5p.pkl"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)

# เราจะแบ่งหน้าจอเป็น 
# 1. sidebar ประกอบด้วยตัวเลือกรูปภาพ
# 2. main page ประกอบด้วยรูปและคำทำนาย

##################################
# sidebar
##################################

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
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)
    if fname is None:
        fname = valid_images[0]

##################################
# main page
##################################

# ใส่ title ของ main page
st.title("Chocolate Chip vs Raisin Cookies")

#function การทำนาย
def predict(img, learn):

    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(img)

    # โชว์ผลการทำนาย
    st.success(f"This is {pred} cookie with the probability of {pred_prob[pred_idx]*100:.02f}%")
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)

# เปิดรูป
img = PILImage.create(fname)

# เรียก function ทำนาย
predict(img, learn_inf)