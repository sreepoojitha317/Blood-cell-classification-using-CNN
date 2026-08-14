# Blood Cell Classification — Simple Streamlit App


import streamlit as st
from PIL import Image

from src.predict import predict_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Blood Cell Classification",
    page_icon="🩸",
    layout="centered"
)


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #1e3a8a;
}

.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 30px;
}

.result-box {
    background-color: #eef4ff;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #cbd5e1;
}

.result-class {
    font-size: 30px;
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 8px;
}

.confidence {
    font-size: 20px;
    font-weight: 600;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩸 Blood Cell AI")

    st.write(
        "Microscopic blood cell image classification "
        "using a CNN model."
    )

    st.divider()

    st.subheader("🔬 Classes")

    st.write("🟣 Eosinophil")
    st.write("🔵 Lymphocyte")
    st.write("🟢 Monocyte")
    st.write("🔴 Neutrophil")

    st.divider()

    st.subheader("🤖 Model")

    st.write("Architecture: CNN")
    st.write("Input: 128 × 128")
    st.write("Classes: 4")
    st.write("Framework: TensorFlow / Keras")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩸 Blood Cell Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered microscopic blood cell image classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.info(
    "🔍 How it works\n\n"
    "Upload a microscopic blood cell image. "
    "The trained CNN model analyzes the image and "
    "predicts one of four blood cell types along "
    "with its confidence score."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Blood Cell Image")

uploaded_file = st.file_uploader(
    "Choose a blood cell image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        caption="Input Blood Cell Image",
        width=400
    )

    st.divider()

    if st.button(
        "🔬 Predict Blood Cell",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            temp_path = "temp_uploaded_image.jpg"

            image.convert("RGB").save(temp_path)

            predicted_class, confidence = predict_image(
                temp_path
            )

        st.success("Prediction completed successfully!")

        st.subheader("📊 Prediction Result")

        # Prediction name
        st.markdown("**Predicted Blood Cell**")

        st.markdown(
            f"# 🩸 {predicted_class}"
        )

        # Confidence
        st.markdown(
            f"### Confidence: {confidence:.2f}%"
        )

        
# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ Important: This application is an AI-based "
    "classification project for educational and "
    "demonstration purposes. Model confidence does not "
    "guarantee correct classification and the system "
    "should not be used as a medical diagnostic tool."
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🩸 Blood Cell Classification | "
    "CNN-based Image Classification | "
    "TensorFlow / Keras"
)

