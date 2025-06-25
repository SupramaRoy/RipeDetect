import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Fruit Ripeness Detector")
st.title("🍎 Fruit Ripeness Detector")
st.write("Upload a fruit image to check if it's ripe or raw.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Uploaded Image", use_container_width=True)


        model = load_model("fruit_modelnew.h5")
        img = image.resize((224,224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_names = ["Raw", "Ripe"]
        predicted_label = class_names[np.argmax(prediction)]

        st.success(f"📊 Prediction: **{predicted_label}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📸 Please upload an image to begin.")
