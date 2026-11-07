from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ------------------------------------------------------------------
# 1. Page setup
# ------------------------------------------------------------------
st.set_page_config(
                    page_title="Deepfake Detection",
                    page_icon="🔍",
                    layout="wide",
                    initial_sidebar_state="expanded"
                    )

# ------------------------------------------------------------------
# 2. Paths, model files and class names
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"


MODEL_FILES = {
    "real_fake": {
        "Custom CNN":      MODELS_DIR / "real_fake" / "custom_cnn_model_real_fake.pth",
        "ResNet50":        MODELS_DIR / "real_fake" / "resnet50_model_real_fake.pth",
        "VGG16":           MODELS_DIR / "real_fake" / "vgg16_model_real_fake.pth",
        "EfficientNet-B0": MODELS_DIR / "real_fake" / "efficientnet_b0_model_real_fake.pth",
        "ViT-B/16":        MODELS_DIR / "real_fake" / "vit_model_real_fake.pth",
    },
    "gender": {
        "Custom CNN":      MODELS_DIR / "gender" / "custom_cnn_model_gender.pth",
        "ResNet50":        MODELS_DIR / "gender" / "resnet50_model_gender.pth",
        "VGG16":           MODELS_DIR / "gender" / "vgg16_model_gender.pth",
        "EfficientNet-B0": MODELS_DIR / "gender" / "efficientnet_b0_model_gender.pth",
        "ViT-B/16":        MODELS_DIR / "gender" / "vit_model_gender.pth",
    },
    "age_group": {
        "Custom CNN":      MODELS_DIR / "age_group" / "custom_cnn_age_group.pth",
        "ResNet50":        MODELS_DIR / "age_group" / "resnet50_age_group.pth",
        "VGG16":           MODELS_DIR / "age_group" / "vgg16_age_group.pth",
        "EfficientNet-B0": MODELS_DIR / "age_group" / "efficientnet_b0_age_group.pth",
        "ViT-B/16":        MODELS_DIR / "age_group" / "vit_b16_age_group.pth",
    },
}


SELECTED_ARCH = {
                "real_fake": "ResNet50",
                "gender": "ViT-B/16",
                "age_group": "ResNet50"
                }

TASK_TITLES = {"real_fake": "Real / Fake", "gender": "Gender", "age_group": "Age group"}


CLASSES = {
    "real_fake": ["FAKE", "REAL"],                                           # fake=0, real=1
    "gender": ["Female", "Male", "Unknown"],                                 # female=0, male=1, unknown=2
    "age_group": ["18-25", "26-35", "36-50", "50+", "not_applicable", "under_18"],
}
AGE_DISPLAY = {"not_applicable": "Not applicable", "under_18": "Under 18"}

ARCHITECTURES = list(MODEL_FILES["real_fake"].keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# 3. Model architectures (must match the training notebooks exactly)
# ------------------------------------------------------------------
class CNN(nn.Module):
    """Custom CNN from the notebooks (dropout rates only matter during training)."""

    def __init__(self, num_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.block1 = block(3, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)
        self.block4 = block(128, 256)
        self.block5 = block(256, 512)
        self.dropout_conv = nn.Dropout(0.1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.dropout_conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


def head(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(256, num_classes))


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "Custom CNN":
        return CNN(num_classes)

    if arch == "ResNet50":
        m = models.resnet50(weights=None)
        m.fc = head(m.fc.in_features, num_classes)                              # type: ignore
        return m

    if arch == "VGG16":
        m = models.vgg16(weights=None)
        m.classifier = head(512 * 7 * 7, num_classes)                           # type: ignore
        return m

    if arch == "EfficientNet-B0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features                                      # type: ignore
        m.classifier = nn.Sequential(                                           # type: ignore
            nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.5),  # type: ignore
            nn.Linear(256, num_classes))
        return m

    if arch == "ViT-B/16":
        m = models.vit_b_16(weights=None)
        m.heads = head(m.heads[0].in_features, num_classes)                     # type: ignore
        return m

    raise ValueError(f"Unknown architecture: {arch}")


@st.cache_resource
def load_model(task: str, arch: str):
    """Returns (model, error_message). Exactly one of them is None."""
    path = MODEL_FILES[task][arch]
    if not path.exists():
        print(f"[model error] file not found: {path}")
        return None, "This model is currently unavailable."
    try:
        model = build_model(arch, len(CLASSES[task]))
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, None
    except Exception as e:  # wrong architecture / corrupted file
        print(f"[model error] could not load {path.name} as {arch}: {e}")
        return None, "This model is currently unavailable."


# ------------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------------
# Same evaluation transform used in all three training notebooks.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# @st.cache_data
# def load_fig(name: str):
#     """Load a saved plotly figure from /models (.json preferred, else .pkl)."""
#     json_path = MODELS_DIR / f"{name}.json"
#     pkl_path = MODELS_DIR / f"{name}.pkl"
#     if json_path.exists():
#         return pio.read_json(json_path)
#     if pkl_path.exists():
#         return joblib.load(pkl_path)
#     return None


# def show_fig(name: str):
#     fig = load_fig(name)
#     if fig is None:
#         st.warning(f"Figure not found: {name}")
#     else:
#         st.plotly_chart(fig, use_container_width=True)


def predict_probs(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Softmax probabilities for one image."""
    with torch.no_grad():
        logits = model(input_tensor)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()



# ------------------------------------------------------------------
# 5. Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("Deepfake Detection")
page = st.sidebar.radio("Content", ["Home", "EDA", "Image Prediction"])


# ------------------------------------------------------------------
# 6. Home
# ------------------------------------------------------------------
if page == "Home":
    st.header("🔍 Deepfake Detection using Deep Learning")
    st.write(
        "This app classifies a human face image as **REAL** or **FAKE**, explains the decision "
        "with **Grad-CAM** heatmaps, and estimates the person's **gender** and **age group**."
    )

    st.subheader("Project Overview")
    st.markdown(
        """
        - **Dataset:** 5,114 face images (real and AI-generated) after removing failed downloads and
          duplicates, split into train (3,581) / validation (765) / test (768) with stratification on the label.
        - **Models compared (for each task):** Custom CNN, ResNet50, VGG16, EfficientNet-B0, Vision Transformer (ViT-B/16).
        - **Three tasks:** Real vs Fake, Gender (female / male) and Age group
          (under 18, 18-25, 26-35, 36-50, 50+, not applicable).
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
                            "Correlation Analysis",
                            "Confusion Matrix",
                            "ROC curve",
                            "Grad-CAM",
                        ],
                        )

    if selection is None:
        st.info("Select a visualization above.")

    elif selection == "Real vs Fake Image":
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                real_fake_count_before_cleaning_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\real_vs_fake_image_count_before_cleaning.pkl")
                st.plotly_chart(real_fake_count_before_cleaning_fig, use_container_width=True)


        with col2:
            with st.container():
                real_fake_count_after_cleaning_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\real_vs_fake_image_count_after_cleaning.pkl")
                st.plotly_chart(real_fake_count_after_cleaning_fig, use_container_width=True)


    elif selection == "Age Distribution":
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                age_dist_before_cleaning_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\age_distribution_before_cleaning.pkl")
                st.plotly_chart(age_dist_before_cleaning_fig, use_container_width=True)

            age_group_dist_real_and_fake_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\age_group_dist_real_and_fake.pkl")
            st.plotly_chart(age_group_dist_real_and_fake_fig, use_container_width=True )
            
        with col2:
            with st.container():
                age_dist_after_cleaning_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\age_distribution_after_cleaning.pkl")
                st.plotly_chart(age_dist_after_cleaning_fig, use_container_width=True)


    elif selection == "Gender Distribution":
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                gender_dist_before_cleaning_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\gender_distribution_before_cleaning.pkl")
                st.plotly_chart(gender_dist_before_cleaning_fig, use_container_width=True)

            gender_wise_dist_real_and_fake_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\gender_wise_dist_real_and_fake.pkl")
            st.plotly_chart(gender_wise_dist_real_and_fake_fig, use_container_width=True )

        with col2:
            with st.container():
                gender_dist_after_cleaning_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\gender_distribution_after_cleaning.pkl")
                st.plotly_chart(gender_dist_after_cleaning_fig, use_container_width=True)


    elif selection == "Visual Sample Inspection":
                visual_sample_insp_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\visual_sample_inspection.pkl")
                st.plotly_chart(visual_sample_insp_fig, use_container_width=True)


    elif selection == "Correlation Analysis":
            correlation_analysis_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\eda\correlation_matrixs.pkl")
            st.plotly_chart(correlation_analysis_fig, use_container_width=True)

    elif selection == "Confusion Matrix":
        col1, col2, col3 = st.columns(3)
        with col1:
                real_fake_resnet50_train_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\real_fake\confusion_matrix_real_fake_ResNet50_train.pkl")
                st.plotly_chart(real_fake_resnet50_train_fig, use_container_width=True)
                real_fake_resnet50_test_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\real_fake\confusion_matrix_real_fake_ResNet50_test.pkl")
                st.plotly_chart(real_fake_resnet50_test_fig, use_container_width=True)
                real_fake_resnet50_val_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\real_fake\confusion_matrix_real_fake_ResNet50_val.pkl")
                st.plotly_chart(real_fake_resnet50_val_fig, use_container_width=True)
        with col2:
                gender_resnet50_train_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\gender\confusion_matrix_gender_resnet50_train.pkl")
                st.plotly_chart(gender_resnet50_train_fig, use_container_width=True)
                gender_resnet50_test_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\gender\confusion_matrix_gender_resnet50_test.pkl")
                st.plotly_chart(gender_resnet50_test_fig, use_container_width=True)
                gender_resnet50_val_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\gender\confusion_matrix_gender_resnet50_val.pkl")
                st.plotly_chart(gender_resnet50_val_fig, use_container_width=True)

        with col3:
                age_group_vit_b16_train_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\age_group\confusion_matrix_age_group_ViT_B16_train.pkl")
                st.plotly_chart(age_group_vit_b16_train_fig, use_container_width=True)
                age_group_vit_b16_test_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\age_group\confusion_matrix_age_group_ViT_B16_test.pkl")
                st.plotly_chart(age_group_vit_b16_test_fig, use_container_width=True)
                age_group_vit_b16_val_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\age_group\confusion_matrix_age_group_ViT_B16_val.pkl")
                st.plotly_chart(age_group_vit_b16_val_fig, use_container_width=True)


    elif selection == "ROC curve":
        real_fake_roc_curve_resnet50_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\real_fake\real_fake_ROC_Curve_ResNet50.pkl")
        st.plotly_chart(real_fake_roc_curve_resnet50_fig, use_container_width=True)

        gender_roc_curve_resnet50_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\gender\ROC_Curve-ResNet50.pkl")
        st.plotly_chart(gender_roc_curve_resnet50_fig , use_container_width=True)

        age_group_vit_b16_train_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\charts\age_group\ROC_Curve_Age_group-ViT-B16.pkl")
        st.plotly_chart(age_group_vit_b16_train_fig, use_container_width=True)


    elif selection == "Grad-CAM":
        show_fig("grad_cam_fig")


# ------------------------------------------------------------------
# 8. Image Prediction
# ------------------------------------------------------------------
elif page == "Image Prediction":
    st.header("🔍 Deepfake Detection using Deep Learning")
    st.write("Upload a human face image to check whether it is **REAL or FAKE**, "
             "and to estimate the **gender** and **age group**.")

    # Load the three selected models (cached after the first load).
    rf_arch, gender_arch, age_arch = (SELECTED_ARCH["real_fake"], SELECTED_ARCH["gender"],
                                      SELECTED_ARCH["age_group"])
    real_fake_model, rf_err = load_model("real_fake", rf_arch)
    gender_model, gender_err = load_model("gender", gender_arch)
    age_model, age_err = load_model("age_group", age_arch)

    if real_fake_model is None:
        st.error(rf_err)
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open image (fix phone-photo rotation, force RGB)
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        if st.button("🔍 Predict", type="primary"):
            input_tensor = transform(image).unsqueeze(0).to(device)  # type: ignore

            # ---- Real / Fake ------------------------------------------------
            rf_probs = predict_probs(real_fake_model, input_tensor)   # [P(fake), P(real)]
            rf_idx = int(rf_probs.argmax())
            prediction = CLASSES["real_fake"][rf_idx]
            confidence = float(rf_probs[rf_idx]) * 100

            # ---- Gender and age group --------------------------------------
            gender_probs = predict_probs(gender_model, input_tensor) if gender_model is not None else None
            age_probs = predict_probs(age_model, input_tensor) if age_model is not None else None

            with right:
                st.subheader("Prediction Result")
                if prediction == "FAKE":
                    st.error(f"This image looks **FAKE** ({confidence:.2f}% confidence)")
                else:
                    st.success(f"This image looks **REAL** ({confidence:.2f}% confidence)")

                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", prediction)
                c2.metric("Confidence", f"{confidence:.2f}%")
                c3.metric("FAKE / REAL", f"{rf_probs[0] * 100:.0f}% / {rf_probs[1] * 100:.0f}%")

                st.subheader("Estimated Gender and Age Group")
                c1, c2 = st.columns(2)
                with c1:
                    if gender_probs is None:
                        st.warning(gender_err)
                    else:
                        g_idx = int(gender_probs.argmax())
                        st.metric("Gender", CLASSES["gender"][g_idx])
                with c2:
                    if age_probs is None:
                        st.warning(age_err)
                    else:
                        a_idx = int(age_probs.argmax())
                        a_name = CLASSES["age_group"][a_idx]
                        st.metric("Age Group", AGE_DISPLAY.get(a_name, a_name))