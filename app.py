
from pathlib import Path
import sys

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


# PAGE CONFIGURATION
st.set_page_config(
    page_title="MedVision AI - Chest X-Ray Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# PROJECT PATHS
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "chest_xray_cnn.pth"
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from model import ChestXRayCNN


# DEVICE
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# DASHBOARD STYLING
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: #111c30;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .card-heading {
        color: #38bdf8;
        font-size: 13px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .card-value {
        color: #f8fafc;
        font-size: 23px;
        font-weight: bold;
        margin-top: 8px;
    }

    div.stButton > button {
        background-color: #0284c7;
        color: white;
        border-radius: 9px;
        border: none;
        font-weight: bold;
        width: 100%;
    }

    [data-testid="stFileUploader"] {
        background-color: #111c30;
        border: 1px dashed #38bdf8;
        border-radius: 12px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}. "
            "Please run python src/train.py first."
        )

    model = ChestXRayCNN().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# PREPROCESS IMAGE
def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    return transform(image).unsqueeze(0).to(DEVICE)


# SIDEBAR
with st.sidebar:
    st.markdown("## 🩻 MedVision AI")
    st.caption("Chest X-Ray Analysis Dashboard")

    st.divider()

    st.markdown("### Navigation")
    st.write("🏠 Dashboard")
    st.write("🔬 Image Analysis")
    st.write("🧠 CNN Model")

    st.divider()

    st.markdown("### Model Information")
    st.write("Architecture: Custom CNN")
    st.write("Input size: 224 × 224")
    st.write("Classes: Normal / Pneumonia")
    st.write(f"Device: {DEVICE}")

    st.divider()

    st.caption(
        "Educational project only. "
        "Not intended for clinical diagnosis."
    )


# DASHBOARD HEADER
st.markdown(
    '<div class="main-title">Chest X-Ray Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a chest X-ray image to receive an experimental '
    'CNN model prediction.'
    '</div>',
    unsafe_allow_html=True,
)


# OVERVIEW CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-heading">AI Model</div>
            <div class="card-value">Custom CNN</div>
            <p>Deep learning image classifier</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-heading">Image Input</div>
            <div class="card-value">224 × 224</div>
            <p>Resized grayscale image</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-heading">Output Classes</div>
            <div class="card-value">2 Classes</div>
            <p>NORMAL or PNEUMONIA</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# IMAGE UPLOAD AND RESULTS
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("📤 Upload X-Ray Image")

    st.write(
        "Select a chest X-ray image in JPG, JPEG, or PNG format."
    )

    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        key="xray_uploader",
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to start the analysis automatically."
        )


with right:
    st.subheader("🔬 Analysis Results")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-heading">Status</div>
                <div class="card-value">Waiting for Image</div>
                <p>Your prediction will appear here after upload.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption=f"Uploaded image: {uploaded_file.name}",
                width="stretch",
            )

            with st.spinner("Analyzing image..."):
                model = load_model()
                input_tensor = preprocess_image(image)

                with torch.inference_mode():
                    outputs = model(input_tensor)
                    predicted_index = outputs.argmax(dim=1).item()

                prediction = CLASS_NAMES[predicted_index]

            st.markdown("### Prediction")

            if prediction == "PNEUMONIA":
                st.error("Model prediction: PNEUMONIA")
            else:
                st.success("Model prediction: NORMAL")

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-heading">Predicted Class</div>
                    <div class="card-value">{prediction}</div>
                    <p>Output generated by your trained CNN model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.warning(
                "This is an experimental student-project prediction, "
                "not a medical diagnosis. The model is not clinically "
                "validated and may make mistakes. Do not use it to "
                "make medical decisions."
            )

        except FileNotFoundError as error:
            st.error("The trained model file is missing.")
            st.code(str(error))
            st.info(
                "Check that models/chest_xray_cnn.pth exists "
                "inside your project folder."
            )

        except Exception as error:
            st.error("The image could not be analyzed.")
            st.exception(error)


# FOOTER
st.divider()

st.caption(
    "MedVision AI | AI-Powered Medical Image Analysis System "
    "| Built with PyTorch and Streamlit | Educational use only"
)

