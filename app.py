import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model and labels once
model = load_model("./model/cancer_model.h5")
with open("labels.json", "r") as f:
    labels = json.load(f)

# Show model input shape in console/log (for debugging)
st.write(f"Model input shape: {model.input_shape}")

# Extract expected input size from model input shape
# Usually (None, height, width, channels)
_, height, width, channels = model.input_shape

st.title("🩺 Skin Cancer Classification")
st.write("Upload a skin lesion image to predict its category.")

def preprocess_image(img, target_size=(height, width)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            try:
                img_array = preprocess_image(img, target_size=(width, height))
                predictions = model.predict(img_array)
                class_index = np.argmax(predictions)
                confidence = np.max(predictions) * 100

                st.write(f"**Prediction:** {labels[str(class_index)]}")
                st.write(f"**Confidence:** {confidence:.2f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload an image to enable prediction.")
