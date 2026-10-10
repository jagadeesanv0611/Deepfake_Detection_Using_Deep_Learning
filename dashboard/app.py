from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import plotly.io as pio
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms


# ------------------------------------------------------------------
# 1. Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 2. Paths (relative to the project root, works locally and on a server)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "resnet50_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# 3. Load model (cached, loaded only once)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)

    # Same classification head used during training
    model.fc = nn.Sequential(       #type: ignore
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

resnet_model = load_model()


# ------------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_data
def load_fig(name: str):
    """Load a saved plotly figure from /models.

    Accepts a name like 'age_distribution_before_cleaning'. It uses the .json
    version if it exists (smaller, safer), otherwise the .pkl version.
    """
    json_path = MODELS_DIR / f"{name}.json"
    pkl_path = MODELS_DIR / f"{name}.pkl"
    if json_path.exists():
        return pio.read_json(json_path)
    if pkl_path.exists():
        return joblib.load(pkl_path)
    return None


def show_fig(name: str):
    fig = load_fig(name)
    if fig is None:
        st.warning(f"Figure not found: {name}")
    else:
        st.plotly_chart(fig, use_container_width=True)


class BinaryOutputTarget:
    """Grad-CAM target for a single-logit binary model.

    Positive logit = FAKE. For a REAL prediction the sign is flipped so the
    heatmap explains why the image looks REAL.
    """

    def __init__(self, is_fake: bool):
        self.sign = 1 if is_fake else -1

    def __call__(self, model_output):
        return self.sign * model_output[0]


# ------------------------------------------------------------------
# 5. Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("Deepfake Detection")
page = st.sidebar.radio("Navigate", ["Home", "EDA", "Image Prediction"])


# ------------------------------------------------------------------
# 6. Home
# ------------------------------------------------------------------
if page == "Home":
    st.header("🔍 Deepfake Detection using Deep Learning")
    st.write(
        "This app classifies a human face image as **REAL** or **FAKE** using a "
        "fine-tuned **ResNet50** and explains each decision with **Grad-CAM** heatmaps."
    )

    st.subheader("Project Overview")
    st.markdown(
        """
        - **Dataset:** 5,557 face images (real and AI-generated), split into train / validation / test
          (about 70 / 15 / 15) with stratification on the label.
        - **Models compared:** Custom CNN, ResNet50 (transfer learning), Vision Transformer (ViT-B/16).
        - **Deployed model:** ResNet50, with the last block (`layer4`) fine-tuned and a custom classification head.
        - **Explainability:** Grad-CAM highlights the regions that influenced the prediction.
        """
    )


# ------------------------------------------------------------------
# 7. EDA
# ------------------------------------------------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis")

    selection = st.pills(
        "Data Visualization",
        options=[
            "Real vs Fake Image",
            "Age Distribution",
            "Gender Distribution",
            "Visual Sample Inspection",
            "Real and Fake Images",
            "Correlation Analysis",
            "Confusion Matrix",
            "ROC curve",
            "Grad-CAM"
                    ],)

    if selection == "Real vs Fake Image":
        col1, col2 = st.columns(2)
        with col1:
            show_fig("real_vs_fake_image_count_before_cleaning")
        with col2:
            show_fig("real_vs_fake_image_count_after_cleaning")

    elif selection == "Age Distribution":
        col1, col2 = st.columns(2)
        with col1:
            show_fig("age_distribution_before_cleaning")
        with col2:
            show_fig("age_distribution_after_cleaning")
        show_fig("age_group_dist_real_and_fake")

    elif selection == "Gender Distribution":
        col1, col2 = st.columns(2)
        with col1:
            show_fig("gender_distribution_before_cleaning")
        with col2:
            show_fig("gender_distribution_after_cleaning")
        show_fig("gender_wise_dist_real_and_fake")

    elif selection == "Visual Sample Inspection":
        show_fig("visual_sample_inspection")

    elif selection == "Real and Fake Images":
        show_fig("class_wise_real_and_fake")

    elif selection == "Correlation Analysis":
        show_fig("correlation_matrix")
        show_fig("correlation_image_feature_label")
        show_fig("correlation_gender_group_label")
        show_fig("correlation_age_group_label")


    elif selection == "Confusion Matrix":
        show_fig("confu_resnet_train_fig")
        show_fig("confu_resnet_val_fig")
        show_fig("confu_resnet_test_fig")

    elif selection == "ROC curve":
        show_fig("resnet_roc_curve_fig")

    elif selection == "Grad-CAM":
        show_fig("grad_cam_fig")




# ------------------------------------------------------------------
# 8. Image Prediction
# ------------------------------------------------------------------
elif page == "Image Prediction":
    st.header("🔍 Deepfake Detection using Deep Learning")
    st.write("Upload a human face image to check whether it is REAL or FAKE.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open image (fix phone-photo rotation, force RGB)
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, width=500)

        if st.button("🔍 Predict", type="primary"):
            input_tensor = transform(image).unsqueeze(0).to(device) #type: ignore

            # Model prediction
            with torch.no_grad():
                output = resnet_model(input_tensor)
                probability = torch.sigmoid(output).item()  # probability of FAKE

            if probability >= 0.5:
                prediction = "FAKE"
                confidence = probability * 100
            else:
                prediction = "REAL"
                confidence = (1 - probability) * 100

            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", prediction)
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")

            st.write(f"**FAKE Probability:** {probability * 100:.2f}%")
            st.write(f"**REAL Probability:** {(1 - probability) * 100:.2f}%")

            # Grad-CAM
            st.subheader("Grad-CAM Explanation")

            # 'with' removes the hooks afterwards, so repeated predictions
            # do not slow down the app or leak memory
            with GradCAM(
                model=resnet_model,
                target_layers=[resnet_model.layer4[-1]],
            ) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[BinaryOutputTarget(prediction == "FAKE")], #type: ignore
                )[0]

            rgb_image = np.array(image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Image")
                st.image(rgb_image, use_container_width=True)
            with col2:
                st.write("Grad-CAM")
                st.image(visualization, use_container_width=True)

            st.caption(
                f"The heatmap shows the regions that most supported the "
                f"**{prediction}** prediction (red = strongest influence)."
            )
