import splitfolders
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np

# Define input and output folders
input_folder = 'RealWaste'  # Adjust as per your directory structure
output_folder = 'output'

# Split the dataset
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15))

# Define directories
train_dir = 'output/train'
val_dir = 'output/val'
test_dir = 'output/test'

# Image data generator with augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Image data generator for validation and test sets (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'), MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'), Dropout(0.5),
    Dense(9, activation='softmax')  # 9 categories of waste
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Define the checkpoint callback to save the best model based on validation accuracy
checkpoint_filepath = 'best_model.keras'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
callbacks_list = [checkpoint]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=callbacks_list
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
st.write('Test accuracy:', test_acc)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

st.pyplot(plt)

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
