# Deepfake Detection Using Deep Learning:

# Project Overview:
- The focus of this project is to build a deep learning system that detects whether a given face image is REAL or FAKE (deepfake-generated). The task is treated as supervised binary image classification. This project follows an ML pipeline. Three models are trained and compared: a custom CNN built from scratch, a fine-tuned ResNet50 and a fine-tuned Vision Transformer (ViT). They are evaluated with accuracy, precision, recall, F1-score and AUC-ROC to find the most accurate one. The best model, selected based on its test accuracy, is deployed as a Streamlit web app.

# Technology Used:
### Data Preprocessing:
- pandas
- numpy
- tqdm
- opencv-python
- pillow
- path

### Model Building & Training:
- torch
- torchvision
- copy

### Evaluation:
- grad-cam
- matplotlib
- seaborn
- plotly
- scikit-learn

### Deployment:
- requests
- streamlit
- nbformat




# Data Collection:
- Data collection is to gather and organize the raw data in a structured format, making it ready for preprocessing, analysis and machine learning model development.
- The Source file listed with 6,557 records and 5 features which are image_id, image_url, label ( Real or Fake), gender (Male or Female), and age group (18 - 50+).

# Data Cleaning:
- The process of cleaning and preparing raw data can be effectively used for analysis and machine learning.
- Handle missing values (null data) to ensures the dataset is complete and prevents errors during analysis and model training.
- Removing the duplicate image urls.
- Here images downloads are done with the help of image URL's and image id's. where "requests.get(url, timeout=10)" ask server for image and downloaded images are stored in data/image folder and failed one's are saved as Failed Download.csv. The script is runed for three times to get total count. Counts got while running for first time is 2,894 counts, second time is 5,547 and third time is 5,557.

| Item | Value |
|---|---|
| Rows in the original CSV | 6,557 (3,767 FAKE, 2,790 REAL) |
| Images downloaded successfully | 5,557 (84.8%) |
| Failed downloads | 1,000 (dead links, timeouts, connection errors) |
 
- After Downloading only 5,557 images are passed and 1,000 are failed to download, so failed one can be considered as fake. Even though images are downloaded have to check whether any duplicate images are repeated because for model train and analyze their should not be repeated images, then the model will be affected in accuracy. After removing duplicate images, final count is 5114, so, the overall percentage is 78%.

# Data Splitting:
- Before preprocessing data must be split in 70/15/15, it is splitted in stratified way train/test/validation, where the overall combine real and fake counts 5,114 is splitted for train -> 3581 images, val -> 765 images, test -> 768 images keeping the Real/Fake ratio equal in each.


| Split | Images | REAL share |
|---|---|---|
| Train | 3,581 | 50.2% |
| Validation | 765 | 50.2% |
| Test | 768 | 50.2% |

# Data Preprocessing:
- In data preprocessing, image resize concept is used because sizes vary widely: 127 to 1080 px wide and 128 to 2440 px high, so every image has to be resized. According to our project requirement all images are resized as 224 x 224 pixels, convert to a tensor, normalize with the ImageNet mean and standard deviation.
- Augmentation done for training data only: random horizontal flip and random rotation up to 10 degrees. Validation and test images are never augmented.
A custom Dataset loads one image at a time. Dataloader batches them (batch size 32) and shuffles only the training data. No shuffle done in validation and test.

# EDA:
- Exploratory Data Analysis (EDA) is used to summarize, visualize, and understand the dataset.

## Visualize Sample Inspection:
<img width="1000" height="1000" alt="sample_images_visualization" src="https://github.com/user-attachments/assets/0245c7bb-9268-4f11-a383-073227baea45" />
- Here the some of the sample images are shown that how it looks in the image folder.

## Class wise visual comparison between Real and Fake:
<img width="1200" height="1000" alt="class_wise_visual_comparison" src="https://github.com/user-attachments/assets/953b59f4-3906-42b7-8943-c4cec86505ca" />
- The above chart shows what are the real images and fake images.


## Real vs Fake images Before and After:
<img width="1353" height="512" alt="real_vs_fake_before_after" src="https://github.com/user-attachments/assets/ab3eb40a-f794-41ae-9f31-5acce5febef9" />
- Image count, Before cleaning real -> 2,790 (42.5%) and fake -> 3,767 (57.5%) and After cleaning real -> 2,368 (46.3%) and fake -> 2,746 (53.7%). From this we can say more than 1,000 images are fake and some count reduced in real is duplicate.

## Gender Count Before and After:
<img width="1353" height="572" alt="gender_dist_before_after" src="https://github.com/user-attachments/assets/1c791727-74f5-4e57-b847-4ef5b69bf370" />
- Gender count before cleaning Unknown -> 4,507 , Female -> 1,033 & Male -> 1,017,  whereas after cleaning Unknown -> 3,354, Female -> 890 & Male -> 870. From this we can understand moreover counts are affected for both genders.

## Age group count before and after:
<img width="1352" height="570" alt="image" src="https://github.com/user-attachments/assets/8e2fcfe2-bc68-4df7-aec4-07b1f7a6557d" />
<img width="608" height="555" alt="age_group_wise_dist_real_and_fake_images" src="https://github.com/user-attachments/assets/bfbacdf3-ddcb-4835-96f8-ced09a94a862" />

- Age counts before cleaning 18-25 -> 1681, 36-50 -> 1670, 26-35 -> 1650 & 50+ -> 1556, whereas after cleaning 18-25 -> 1318, 36-50 -> 1301, 26-35 -> 1258 & 50+ -> 1237.
- From this picture we're come's to know that real and fake images are somewhat equally placed.

# Correlation Analysis:
- Correlation analysis is made to check whether each feature depends on other feature in a dataset.

## Association between variables:
<img width="602" height="528" alt="correlation_bet_variables" src="https://github.com/user-attachments/assets/b010b47d-0f44-4926-bb8e-46af8e6192f8" />

- From the above chart, we found Gender(63%) and age group (3%) is how much it correlate with Label. By using this we can predict whether those feature affect the label feature.


## Label share with gender and age group:
<img width="598" height="552" alt="label_share_inside_each_gender" src="https://github.com/user-attachments/assets/2199e046-f5f8-4074-8ff5-dfd205896130" />
<img width="596" height="557" alt="label_share_inside_each_age_group" src="https://github.com/user-attachments/assets/06745c5f-8c1e-4278-a91d-be237bb7cfca" />

- From the above chart we understood how the fake and real images are shared equally in age wise and gender wise.


## Correlation between image features and label:
<img width="712" height="591" alt="correlation_image_features_with_label" src="https://github.com/user-attachments/assets/2504e296-c068-4458-894b-563ea04a45a3" />

- From the above chart we understood how the image features width, height, aspect ratio, file size kb and brightness are correlate with label.

# Model Training:
- For this project three models are trained, those are Custom CNN, ResNet50 and ViT.
- The results are,

### Model Performance Comparison

| Model | Metric | Train | Validation | Test |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | Accuracy | 0.9668 | 0.9686 | 0.9688 |
| | Precision | 0.9967 | 0.9974 | 0.9949 |
| | Recall | 0.9412 | 0.9440 | 0.9466 |
| | F1-Score | 0.9682 | 0.9700 | 0.9701 |
| | AUC-ROC | 0.9923 | 0.9909 | 0.9911 |
| **ResNet50** | Accuracy | 0.9994 | 0.9935 | 0.9948 |
| | Precision | 1.0000 | 1.0000 | 0.9951 |
| | Recall | 0.9990 | 0.9878 | 0.9951 |
| | F1-Score | 0.9995 | 0.9939 | 0.9951 |
| | AUC-ROC | 1.0000 | 0.9998 | 0.9998 |
| **ViT** | Accuracy | 0.9989 | 0.9895 | 0.9922 |
| | Precision | 1.0000 | 0.9951 | 0.9975 |
| | Recall | 0.9979 | 0.9854 | 0.9879 |
| | F1-Score | 0.9990 | 0.9902 | 0.9927 |
| | AUC-ROC | 1.0000 | 0.9991 | 0.9992 |

- From this ResNet50 is best when compare with other models results.


# Confusion Matrix:
### Based on testing:
<img width="602" height="553" alt="confusion_matrix_ResNet50_test" src="https://github.com/user-attachments/assets/cf21f7a8-fe6c-453d-80d0-cc7b99886ec0" />

### Based on validation:
<img width="598" height="565" alt="confusion_matrix_ResNet50_val" src="https://github.com/user-attachments/assets/5e6912a5-ca1f-4d28-afa8-f19b91b4e79c" />

### Based on training:
<img width="605" height="562" alt="confusion_matrix_ResNet50_train" src="https://github.com/user-attachments/assets/3d543f07-22c7-4fc6-80a0-efc65c5cf943" />


# Model Deployment:
- Model is deployed using streamlit.
https://deepfakedetectionusingdeeplearning-2y6kp2b7zdvwcgk2nj5hn9.streamlit.app/
