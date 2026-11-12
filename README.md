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
 
- After Downloading only 5,557 images are passed and 1,000 are failed to download, so failed one can be considered as fake. Even though images are downloaded have to check whether any duplicate images are repeated because for model train and analyze their should not be repeated images, then the model will be affected in accuracy. After removing duplicate images, final count is 5088, so, the overall percentage is 78%.

# Data Splitting:
- Before preprocessing data must be split in 70/20/10, it is splitted in stratified way train/test/validation, where the overall combine real and fake counts 5,088 is splitted for train -> 3561 images, val -> 508 images, test -> 1019 images keeping the Real/Fake ratio equal in each.


| Split | Images | REAL share |
|---|---|---|
| Train | 3,561 | 70% |
| Validation | 508 | 10% |
| Test | 1019 | 20% |

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
<img width="1117" height="368" alt="real_vs_fake_before_after" src="https://github.com/user-attachments/assets/92e751f3-032e-48b6-b426-7be1c8dbf04f" />

- Image count, Before cleaning real -> 2,790 (42.5%) and fake -> 3,767 (57.5%) and After cleaning real -> 2,593 (51%) and fake -> 2,495 (49%). From this we can say more than 1,000 images are fake and some count reduced in real is duplicate.

## Gender Count Before and After:
<img width="1114" height="402" alt="gender_dist_before_after" src="https://github.com/user-attachments/assets/0e64cacf-8d4c-412d-8c4e-8e86829a5ad7" />
<img width="550" height="397" alt="gender_wise_dist_real_and_fake_images" src="https://github.com/user-attachments/assets/46dc0869-a668-49ad-a462-a9fc2a5c5003" />


- Gender count before cleaning Unknown -> 4,507 , Female -> 1,033 & Male -> 1,017,  whereas after cleaning Unknown -> 2,612, Female -> 1,453 & Male -> 1,023. From this we can understand moreover counts are affected for both genders.

## Age group count before and after:
<img width="1114" height="408" alt="age_dist_before_after" src="https://github.com/user-attachments/assets/a1d813b6-879b-4547-b7a4-8aa51d4669bd" />
<img width="558" height="402" alt="age_group_wise_dist_real_and_fake_images" src="https://github.com/user-attachments/assets/ce519d3d-1e75-4423-a1ef-76ec8de6af17" />


- Age counts before cleaning 18-25 -> 1681, 36-50 -> 1670, 26-35 -> 1650 & 50+ -> 1556, whereas after cleaning 18-25 -> 330, 36-50 -> 338, 26-35 -> 1,484, 50+ -> 219 & not_applicable -> 2,638.
- From this picture we're come's to know that real and fake images are somewhat equally placed.



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
- For this project five models are trained, those are Custom CNN, EfficientNet-B0, ResNet50, VGG16 & ViT (Vision Transfer)
- The results are,

### Model Performance Comparison
- From these five models,
  - For Real/Fake, ResNet50
  - For Age group & Gender, ViT



# Confusion Matrix:
### Based on testing:
<img width="1118" height="386" alt="confusion_matrix_test" src="https://github.com/user-attachments/assets/79052448-302e-47db-8051-c4eb6d3f5014" />


### Based on validation:
<img width="1110" height="392" alt="confusion_matrix_val" src="https://github.com/user-attachments/assets/94b78298-faca-4961-b367-c905625be563" />


### Based on training:
<img width="1098" height="378" alt="confusion_matrix_train" src="https://github.com/user-attachments/assets/6c59e2e0-a6a3-4fe0-9821-46ca61863e46" />



# Model Deployment:
- Model is deployed using streamlit.
https://deepfakedetectionusingdeeplearning-2y6kp2b7zdvwcgk2nj5hn9.streamlit.app/
