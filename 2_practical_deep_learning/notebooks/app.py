import streamlit as st
from fastai.vision.all import load_learner, PILImage
import pathlib

# Persistencia en memoria
@st.cache_resource
def load_model():
            model_path = pathlib.Path(__file__).parent / 'model.pkl'
            return load_learner(model_path)
learn = load_model()

st.markdown("<h2 style='text-align: center;'>Alligator vs Crocodile</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
            img = PILImage.create(uploaded_file)
            st.image(img, use_container_width=True)

# Forward pass
pred_class, pred_idx, outputs = learn.predict(img)
prob = outputs[pred_idx].item()

st.success(f"**Class:** {pred_class} | **Probability:** {prob:.4f}")
