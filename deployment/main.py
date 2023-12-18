import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

set_background('./images/crop_image.jpg')

# set title with white text
st.title('Eyes on the ground: Crop Damage Classifier')
st.markdown("<style>h1{color: white;}</style>", unsafe_allow_html=True)

# set header with white text
st.header('Please upload an image of your crop')
st.markdown("<style>h2{color: white;}</style>", unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./models/optimized_model')

# load class names
with open('./labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# set paragraph text to white
st.markdown("<style>p{color: white;}</style>", unsafe_allow_html=True)

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification with white text
    
    if class_name == "Good":
        st.write("This crop is exhibiting good growth.")
    else:
        st.write("This crop is exhibiting {} damage.".format(class_name))
    
