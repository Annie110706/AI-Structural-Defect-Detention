import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


# Load the trained AI model

model = tf.keras.models.load_model("model/defect_model.keras")

# Class names (same order used during training)
class_names = ["crack", "no_defect"]


# Streamlit Page Settings

st.set_page_config(
    page_title="Structural Defect Detection",
    page_icon="🏗️",
    layout="centered"
)

# Title

st.title("🏗️ Structural Defect Detection using AI")

st.write(
    "Upload an image of a concrete or masonry surface and "
    "the AI will detect whether a structural crack is present."
)

st.divider()

# Upload Image

uploaded_file = st.file_uploader(
    "📁 Choose an image",
    type=["jpg", "jpeg", "png"]
)

# If Image Uploaded
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    # Detect Button
    if st.button("🔍 Detect Defect"):

        with st.spinner("Analyzing image..."):

            # Image Preprocessing


            # Convert image to RGB
            img = image.convert("RGB")

            # Resize image
            img = img.resize((224, 224))

            # Convert to NumPy array
            img_array = np.array(img)

            # Normalize pixel values
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            
            # Prediction
            
            prediction = model.predict(img_array)

            predicted_index = np.argmax(prediction)

            predicted_class = class_names[predicted_index]

            confidence = np.max(prediction)

        st.divider()

        st.subheader("Prediction Result")

        
        # Show Result
       
        if predicted_class == "crack":
            st.error("🚨 Crack Detected")
        else:
            st.success("✅ No Defect Detected")

        # Confidence
        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        # Raw prediction (optional)
        with st.expander("Prediction Details"):
            st.write("Prediction Vector:", prediction)
            st.write("Predicted Class:", predicted_class)