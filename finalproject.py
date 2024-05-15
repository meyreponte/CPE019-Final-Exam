import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page config
st.set_page_config(page_title="Emerging Technology 2 in CpE", layout="wide")

# details
st.title("Emerging Technology 2 in CpE")
st.markdown("""
**Name:**
- Mark Janssen Valencia
- Meyrazol Reponte

**Course/Section:** CPE019/CPE32S5

**Date Submitted:** May 17, 2024
""")

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('best_model65.77.keras')
    return model

# Define the class names
class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Example images
example_images = {
    'Cardboard': 'RealWaste/Cardboard/Cardboard_1.jpg',
    'Food Organics': 'RealWastes/Food Organics/Food Organics_1.jpg',
    'Glass': 'RealWaste/Glass/Glass_1.jpg',
    'Metal': 'RealWaste/Metal/Metal_1.jpg',
    'Miscellaneous Trash': 'RealWaste/Miscellaneous Trash/Miscellaneous Trash_1.jpg',
    'Paper': 'RealWaste/Paper/Paper_1.jpg',
    'Plastic': 'RealWaste/Plastic/Plastic_1.jpg',
    'Textile Trash': 'RealWaste/Textile Trash/Textile Trash_1.jpg',
    'Vegetation': 'RealWaste/Vegetation/Vegetation_1.jpg'
}
# Displaying example images for each category
st.write("## Example Images by Category")
for label, path in example_images.items():
    image = Image.open(path)
    st.image(image, caption=f'Example of {label}', use_column_width=True)

# App main interface
st.header("Waste Classification")
st.write("Upload an image to classify the type of waste.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

