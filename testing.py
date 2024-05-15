import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('best_model65.77.keras')

# Define the classes of waste
classes = ["Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash", 
           "Paper", "Plastic", "Textile Trash", "Vegetation"]

def classify_image(img):
    img = img.resize((224, 224))  # Resize the image to match the model's expected input
    img = np.array(img)
    img = img / 255.0  # Scale pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Expand dims to add the batch size
    predictions = model.predict(img)
    confidence = np.max(predictions)  # Confidence of the prediction
    predicted_class = classes[np.argmax(predictions)]
    return predicted_class, confidence

def main():
    st.title("Waste Classification App")
    st.write("This app classifies different types of waste into categories such as Textile Trash, Plastic, etc.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label, confidence = classify_image(image)
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()
