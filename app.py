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
from git import Repo
import shutil

#import streamlit มาในชื่อ st เพื่อใช้ในการสร้าง user interface
import streamlit as st

# clone github repository
if not os.path.exists('/app/dog-breed-classification/Dog-Breed-Classification-Images/images/test/099_Siberian_Husky/n02110185_120.jpg'):
    if not os.path.exists('/app/dog-breed-classification/Dog-Breed-Classification-Images'):
        os.mkdir('/app/dog-breed-classification/Dog-Breed-Classification-Images')
    for root, dirs, files in os.walk('/app/dog-breed-classification/Dog-Breed-Classification-Images'): #เคลียร์ folder
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    Repo.clone_from('https://github.com/PakinDioxide/Dog-Breed-Classification-Images.git', '/app/dog-breed-classification/Dog-Breed-Classification-Images')

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
# st.write(os.listdir('/app/dog-breed-classification/Dog-Breed-Classification-Images'))

##################################
# sidebar
##################################

#function การทำนาย
def predict(img, learn):
    # ย่อขนาดรูป
    pimg = img.resize([224,224])
    
    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(pimg)
        
    pred = pred.split('_')[1:]
    
    if pred[-1] == 'Dog':
        pred = ' '.join(pred[:len(pred)-1])
    else:
        pred = ' '.join(pred)

    # โชว์ผลการทำนาย
    st.success(f'This is "{pred} Dog" with the probability of {pred_prob[pred_idx]*100:.02f}%')
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)
    
    st.balloons()

# ใส่ title ของ sidebar
st.sidebar.write('# Upload a dog image to classify!')

# radio button สำหรับเลือกว่าจะทำนายรูปจาก validation set หรือ upload รูปเอง
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image', 'Take a photo'])
# โหลดรูปจาก validation set แล้ว shuffle
valid_images = glob.glob('/app/dog-breed-classification/Dog-Breed-Classification-Images/images/test/*/*')
# shuffle(valid_images)
for i in range(len(valid_images)):
    k = str(valid_images[i])
    k.replace('/app/dog-breed-classification/Dog-Breed-Classification-Images/images/test/', '')
    valid_images[i] = k
    st.write(k)

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)
    
    # เปิดรูป
    img = Image.open(f'/app/dog-breed-classification/Dog-Breed-Classification-Images/images/test/{fname}')

    st.sidebar.image(img, f'Is this the image you want to predict?', use_column_width=True)

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
        
        st.sidebar.image(img, f'Is this the image you want to predict?', use_column_width=True)

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

            st.sidebar.image(img, 'Is this the image you want to predict?', use_column_width=True)

            if st.sidebar.button("Predict Now!"):
                # เรียก function ทำนาย
                predict(img, learn_inf)
