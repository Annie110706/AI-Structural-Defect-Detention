import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model/defect_model.keras")

# Class names (same order as during training)
class_names = ['crack', 'no_defect']

# Path of the image to test
img_path = "test_images/test.jpg"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Prediction:", predicted_class)
print("Confidence:", confidence)