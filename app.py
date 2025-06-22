import streamlit as st
import torch
from generator_model import load_generator, generate_digit_images
import numpy as np

st.set_page_config(page_title="Handwritten Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator (0–9)")

digit = st.selectbox("Select a digit to generate:", list(range(10)))

if st.button("Generate Images"):
    model = load_generator()
    images = generate_digit_images(model, digit, num_images=5)

    st.write(f"Generated images for digit: {digit}")
    cols = st.columns(5)

    for i, img in enumerate(images):
        # Convert from [-1, 1] → [0, 1] and clip
        img_np = img.squeeze().detach().cpu().numpy()
        img_np = ((img_np + 1) / 2.0).clip(0, 1).astype(np.float32)
        cols[i].image(img_np, width=100, caption=f"{digit}")
