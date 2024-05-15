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

# Example images for each class (dummy paths)
example_images = {
    'Cardboard': 'images/cardboard.jpg',
    'Food Organics': 'images/food_organics.jpg',
    'Glass': 'images/glass.jpg',
    'Metal': 'images/metal.jpg',
    'Miscellaneous Trash': 'images/misc_trash.jpg',
    'Paper': 'images/paper.jpg',
    'Plastic': 'images/plastic.jpg',
    'Textile Trash': 'images/textile_trash.jpg',
    'Vegetation': 'images/vegetation.jpg'
}

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

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display example image for the predicted class
    example_image_path = example_images[predicted_class]
    example_image = Image.open(example_image_path)
    st.image(example_image, caption=f'Example of {predicted_class}', use_column_width=True)
