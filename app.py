import numpy as np
import streamlit as st
from PIL import Image

from predict import load_class_names, load_trained_model, predict_image

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")

st.title("Traffic Sign Recognition Using CNN")
st.subheader("Deep Learning Based Traffic Sign Classification")
st.write("Upload a traffic sign image and the trained CNN will classify it "
         "into one of the 43 GTSRB classes.")


@st.cache_resource
def get_model():
    return load_trained_model()


try:
    model = get_model()
    class_names = load_class_names()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Upload traffic sign image", type=["png", "jpg", "jpeg", "ppm"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=250)

    if st.button("Predict", type="primary"):
        results = predict_image(model, np.array(image), class_names, top_k=3)
        _, name, conf = results[0]

        st.success(f"Predicted traffic sign: **{name}**")
        st.metric("Confidence", f"{conf:.2f}%")

        st.markdown("#### Top 3 predictions")
        for _, n, c in results:
            st.write(f"{n} — {c:.2f}%")
            st.progress(min(int(c), 100))

st.caption("Model: 3-block CNN trained on GTSRB at 32×32 resolution.")
