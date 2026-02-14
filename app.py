import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from fastai.vision.all import PILImage
from PIL import Image
import os
import glob

# 1. THE LABELS
CLASSES = [
    '000_Chihuahua', '001_Japanese_Spaniel', '002_Maltese_Dog', '003_Pekinese', '004_Shih-Tzu',
    '005_Blenheim_Spaniel', '006_Papillon', '007_Toy_Terrier', '008_Rhodesian_Ridgeback', '009_Afghan_Hound',
    '010_Basset', '011_Beagle', '012_Bloodhound', '013_Bluetick', '014_Black-and-tan_Coonhound',
    '015_Walker_Hound', '016_English_Foxhound', '017_Redbone', '018_Borzoi', '019_Irish_Wolfhound',
    '020_Italian_Greyhound', '021_Whippet', '022_Ibizan_Hound', '023_Norwegian_Elkhound', '024_Otterhound',
    '025_Saluki', '026_Scottish_Deerhound', '027_Weimaraner', '028_Staffordshire_Bullterrier', '029_American_Staffordshire_Terrier',
    '030_Bedlington_Terrier', '031_Border_Terrier', '032_Kerry_Blue_Terrier', '033_Irish_Terrier', '034_Norfolk_Terrier',
    '035_Norwich_Terrier', '036_Yorkshire_Terrier', '037_Wire-haired_Fox_Terrier', '038_Lakeland_Terrier', '039_Sealyham_Terrier',
    '040_Airedale', '041_Cairn', '042_Australian_Terrier', '043_Dandie_Dinmont', '044_Boston_Bull',
    '045_Miniature_Schnauzer', '046_Giant_Schnauzer', '047_Standard_Schnauzer', '048_Scotch_Terrier', '049_Tibetan_Terrier',
    '050_Silky_Terrier', '051_Soft-coated_Wheaten_Terrier', '052_West_Highland_White_Terrier', '053_Lhasa', '054_Flat-coated_Retriever',
    '055_Curly-coated_Retriever', '056_Golden_Retriever', '057_Labrador_Retriever', '058_Chesapeake_Bay_Retriever', '059_German_Short-haired_Pointer',
    '060_Vizsla', '061_English_Setter', '062_Irish_Setter', '063_Gordon_Setter', '064_Brittany_Spaniel',
    '065_Clumber', '066_English_Springer', '067_Welsh_Springer_Spaniel', '068_Cocker_Spaniel', '069_Sussex_Spaniel',
    '070_Irish_Water_Spaniel', '071_Kuvasz', '072_Schipperke', '073_Groenendael', '074_Malinois',
    '075_Briard', '076_Kelpie', '077_Komondor', '078_Old_English_Sheepdog', '079_Shetland_Sheepdog',
    '080_Collie', '081_Border_Collie', '082_Bouvier_Des_Flandres', '083_Rottweiler', '084_German_Shepherd',
    '085_Doberman', '086_Miniature_Pinscher', '087_Greater_Swiss_Mountain_Dog', '088_Bernese_Mountain_Dog', '089_Appenzeller',
    '090_EntleBucher', '091_Boxer', '092_Bull_Mastiff', '093_Tibetan_Mastiff', '094_French_Bulldog',
    '095_Great_Dane', '096_Saint_Bernard', '097_Eskimo_Dog', '098_Malamute', '099_Siberian_Husky',
    '100_Affenpinscher', '101_Basenji', '102_Pug', '103_Leonberg', '104_Newfoundland',
    '105_Great_Pyrenees', '106_Samoyed', '107_Pomeranian', '108_Chow', '109_Keeshond',
    '110_Brabancon_Griffon', '111_Pembroke', '112_Cardigan', '113_Toy_Poodle', '114_Miniature_Poodle',
    '115_Standard_Poodle', '116_Mexican_Hairless', '117_Dingo', '118_Dhole', '119_African_Hunting_Dog'
]

# 2. MODEL LOADING (RECONSTRUCT ARCHITECTURE)
@st.cache_resource
def load_weights_only_model():
    # 1. Reconstruct the standard ResNet50 architecture
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    # Match the 120 dog breed classes
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # 2. Path handling for Streamlit Cloud
    weight_path = 'models/resnet50_weights.pth'
    if not os.path.exists(weight_path):
        weight_path = '/mount/src/dog-breed-classification/models/resnet50_weights.pth'
    
    # 3. Load the raw state_dict from your LFS upload
    # map_location='cpu' is essential for Streamlit servers
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    
    # 4. THE CLEANING LOGIC (The "FastAI to PyTorch" Bridge)
    new_state_dict = {}
    
    # We need to find which layer in the FastAI 'head' is the actual final Linear layer
    # Usually, it is '1.8.weight' or '1.6.weight'
    head_keys = [k for k in state_dict.keys() if k.startswith("1.")]
    last_layer_prefix = ""
    if head_keys:
        # Sort and pick the last one (the linear output layer)
        last_layer_prefix = ".".join(head_keys[-1].split(".")[:2]) + "."

    for key, value in state_dict.items():
        if key.startswith("0."):
            # Strip '0.' to map FastAI body weights to standard ResNet layers
            new_key = key.replace("0.", "", 1)
            new_state_dict[new_key] = value
        elif last_layer_prefix and key.startswith(last_layer_prefix):
            # Map the specific FastAI output layer to standard 'fc'
            new_key = key.replace(last_layer_prefix, "fc.", 1)
            new_state_dict[new_key] = value
    
    # 5. Load weights into the architecture
    # strict=False is used because FastAI adds extra BatchNorm layers 
    # that standard ResNet50 doesn't have in its final head.
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

learn_inf = load_weights_only_model()

# 3. PREDICTION LOGIC
def predict(img, model):
    img = img.convert("RGB").resize((224, 224)) # Standard ResNet size
    # Convert to Tensor
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, idx = torch.max(probs, 0)
    
    # Clean up class name for display
    raw_name = CLASSES[idx.item()]
    display_name = ' '.join(raw_name.split('_')[1:])

    st.success(f'This is a "{display_name}" with {conf*100:.02f}% confidence!')
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



