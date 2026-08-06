import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import csv 
from datetime import datetime

from pdf_report import generate_report


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
# ==========================
# Sidebar
# ==========================

st.sidebar.title("📋 Project Info")

st.sidebar.write("""
### AI Structural Defect Detection

This application uses a Deep Learning model to detect structural cracks from images.

**Model:** MobileNetV2

**Classes**
- Crack
- No Defect

**Input Size**
224 × 224 pixels
""")

st.sidebar.divider()

st.sidebar.success("Developed using Python, TensorFlow & Streamlit")
# Title

st.title("🏗️ Structural Defect Detection using AI")
st.caption("AI-powered concrete crack detection using Deep Learning")

st.write(
    "Upload an image of a concrete or masonry surface and "
    "the AI will detect whether a structural crack is present."
)

st.divider()
st.info("📌 Supported file types: JPG, JPEG, PNG")

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
    # Create two columns
    col1, col2 = st.columns([1, 1])

# Left Column
    with col1:
        st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

# Right Column
    with col2:
        st.subheader("Prediction Result")
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

            # Save Prediction history
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M")

            with open("prediction_history.csv","a",
                      newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    current_time,
                    predicted_class,
                    f"{confidence:.2%}"
                ])
            

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
        st.progress(float(confidence))

        #Generate PDF report
        pdf_file = generate_report(predicted_class,confidence)

        with open(pdf_file, "rb")as file:
            st.download_button(
                label="📥 Download Inspection Report",
                data=file,
                file_name="Structural_Defect_Report.pdf",
                mime="application/pdf"
            )
        with st.expander("📊 Prediction Details"):

            st.write("### Class Probabilities")

        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i]:.2%}")

    # Create DataFrame
        df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction[0]
        })

        st.bar_chart(df.set_index("Class"))

        st.write("---")
        st.write("Predicted Class:", predicted_class)
        st.write("---")
        st.subheader("Prediction History")
        history = pd.read_csv("prediction_history.csv")
        st.dataframe(history,use_container_width=True)
        if st.button("Clear History"):
            history.iloc[0:0].to_csv("predicition_history.csv", index=False)
            st.success("Prediction history cleared successfully!")
            st.rerun()