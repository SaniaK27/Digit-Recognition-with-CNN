import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()

st.title("🔢 MNIST Digit Recognizer")
st.write("Draw a digit between 0 and 9 in the box below!")


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


if canvas_result.image_data is not None:
    
    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img).convert('L') 
    
    # MNIST models are trained on 28x28 images
    img = img.resize((28, 28))
    
    
    img_array = np.array(img) / 255.0      # Normalize
    img_array = img_array.reshape(1, 28, 28, 1) # Reshape for CNN input

    if st.button('Predict Digit'):
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.header(f"Result: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2%}")
