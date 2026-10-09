
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import joblib


# Page Setup:
st.set_page_config( page_title="Deepfake Detection", page_icon="🔍", layout="wide",
                    initial_sidebar_state="expanded" )

# 2. Device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )


# 3. Load ResNet50 Model
@st.cache_resource
def load_model():

    # Create ResNet50
    model = models.resnet50(weights=None)

    # Same classification head used during training
    model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\resnet50_model.pth",
            map_location=device ) )
    
    model = model.to(device)
    model.eval()
    return model

resnet_model = load_model()



# 4. Image Transformation
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) ])


# 5. Streamlit Title
st.sidebar.title("Content")
page = st.sidebar.radio( "Content", ["Home","EDA","Image Prediction"] )



if page == "Home":
        st.header("🔍 Deepfake Detection using Deep Learning")



elif page == "EDA":
        st.header("Exploratory Data Analysis")

        selection = st.pills("Data Visualization", 
        options=["Real vs Fake Image", "Age Distribution", "Gender Distribution",
                 "Visual Sample Inspection", "Real and Fake Images", "Confusion Matrix", "ROC curve" ]
                    )

        
        if selection == "Age Distribution":
              col1, col2 = st.columns([1,1])
              with col1:
                    age_distribution_before_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\age_distribution_before_cleaning.pkl")
                    st.plotly_chart(age_distribution_before_fig, use_container_width=True)

              with col2:
                    age_distribution_after_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\age_distribution_after_cleaning.pkl")
                    st.plotly_chart(age_distribution_after_fig, use_container_width=True)

              col3 = st.columns(1)[0]
              with col3:
                    age_group_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\age_group_dist_real_and_fake.pkl")
                    st.plotly_chart(age_group_dist_fig, use_container_width=True)



        elif selection == "Real vs Fake Image":
              col1, col2 = st.columns([1,1])
              with col1:
                    real_vs_fake_before_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\real_vs_fake_image_count_before_cleaning.pkl")
                    st.plotly_chart(real_vs_fake_before_fig, use_container_width=True)

              with col2:
                    real_vs_fake_after_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\real_vs_fake_image_count_after_cleaning.pkl")
                    st.plotly_chart(real_vs_fake_after_fig, use_container_width=True)


        elif selection == "Gender Distribution":
              col1, col2 = st.columns([1,1])
              with col1:
                    gender_dist_before_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\gender_distribution_before_cleaning.pkl")
                    st.plotly_chart(gender_dist_before_fig, use_container_width=True)

              with col2:
                    gender_dist_after_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\gender_distribution_after_cleaning.pkl")
                    st.plotly_chart(gender_dist_after_fig, use_container_width=True)

              col3 = st.columns(1)[0]
              with col3:
                    gender_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\gender_wise_dist_real_and_fake.pkl")
                    st.plotly_chart(gender_dist_fig, use_container_width=True)


        elif selection == "Visual Sample Inspection":
                sample_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\visual_sample_inspection.pkl")
                st.plotly_chart(sample_fig, use_container_width=True)
        

        elif selection == "Real and Fake Images":
                real_and_fake_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\class_wise_real_and_fake.pkl")
                st.plotly_chart(real_and_fake_fig, use_container_width=True)

            
        elif selection == "Confusion Matrix":
                cm_train_resnet_50_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\confu_resnet_train_fig.pkl")
                st.plotly_chart(cm_train_resnet_50_fig, use_container_width=True)

                cm_test_resnet_50_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\confu_resnet_test_fig.pkl")
                st.plotly_chart(cm_test_resnet_50_fig, use_container_width=True)


        elif selection == "ROC curve":
                roc_curve_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\my_projects\Deep_Fake_project_final\models\resnet_roc_curve_fig.pkl")
                st.plotly_chart(roc_curve_fig, use_container_width=True)




elif page == "Image Prediction":

        st.header("🔍 Deepfake Detection using Deep Learning")
        st.write( "Upload a human face image to check whether it is REAL or FAKE." )

        # 6. Upload Image
        uploaded_file = st.file_uploader( "Upload an image",  type=["jpg", "jpeg", "png"] )

        # 7. Upload Image and Prediction Button
        if uploaded_file is not None:

            # Open image
            image = Image.open(uploaded_file).convert("RGB")

            # Display uploaded image
            st.subheader("Uploaded Image")
            st.image( image, width=400 )

            # Prediction Button
            predict_button = st.button( "🔍 Predict", type="primary" )


            # 8. Prediction
            if predict_button:

                # Convert image for model
                input_tensor = transform( image ).unsqueeze(0).to(device)

                # Model prediction
                with torch.no_grad():
                    output = resnet_model( input_tensor )
                    probability = torch.sigmoid( output ).item()


                # Convert probability to prediction
                if probability >= 0.5:
                    prediction = "FAKE"
                    confidence = probability * 100
                else:
                    prediction = "REAL"
                    confidence = (1 - probability) * 100


                # 9. Display Prediction
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric( "Prediction", prediction )
                with col2:
                    st.metric( "Confidence", f"{confidence:.2f}%" )


                # Extra probability information
                st.write( f"**FAKE Probability:** {probability * 100:.2f}%" )

                # 10. Grad-CAM
                st.subheader("Grad-CAM Explanation")

                # Target last convolutional layer
                target_layer = resnet_model.layer4[-1]


                # Create Grad-CAM
                cam = GradCAM( model=resnet_model, target_layers=[target_layer] )

                # Binary classification target
                class BinaryOutputTarget:
                    def __call__(self, model_output):
                        return model_output[0]

                targets = [ BinaryOutputTarget() ]


                # Generate Grad-CAM
                grayscale_cam = cam( input_tensor=input_tensor, targets=targets )[0]


                # Convert image to NumPy
                rgb_image = np.array( image.resize((224, 224)) ) / 255.0


                # Create Grad-CAM visualization
                visualization = show_cam_on_image( rgb_image, grayscale_cam, use_rgb=True )


                # 11. Display Original + Grad-CAM
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Image")
                    st.image( rgb_image, use_container_width=True )

                with col2:
                    st.write("Grad-CAM")
                    st.image( visualization,  use_container_width=True )
