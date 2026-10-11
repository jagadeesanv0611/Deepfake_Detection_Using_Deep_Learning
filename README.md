# Deepfake Detection Using Deep Learning:

# Project Overview:
- The focus of this project is to build a deep learning system that detects whether a given face image is REAL or FAKE (deepfake-generated). The task is treated as supervised binary image classification. This project follows an ML pipeline. Three models are trained and compared: a custom CNN built from scratch, a fine-tuned ResNet50 and a fine-tuned Vision Transformer (ViT). They are evaluated with accuracy, precision, recall, F1-score and AUC-ROC to find the most accurate one. The best model, selected based on its test accuracy, is deployed as a Streamlit web app.

# Technology Used:
- pandas
- numpy
- nbformat
- matplotlib
- seaborn
- plotly
- scikit-learn
- pillow
- requests
- tqdm
- streamlit
- opencv-python
- path
- grad-cam
- torch
- torchvision
- copy

# Data Collection:
- Data collection is to gather and organize the raw data in a structured format, making it ready for preprocessing, analysis and machine learning model development.
- The Source file listed with 6,557 records and 5 features which are image_id, image_url, label ( Real or Fake), gender (Male or Female), and age group (18 - 50+).

# Data Cleaning:
- The process of cleaning and preparing raw data can be effectively used for analysis and machine learning.
- Handle missing values (null data) to ensures the dataset is complete and prevents errors during analysis and model training.
- Removing the duplicate image urls.
- Here images downloads are done with the help of image URL's and image id's. where "requests.get(url, timeout=10)" ask server for image and downloaded images are stored in data/image folder and failed one's are saved as Failed Download.csv. The script is runed for three times to get total count. Counts got while running for first time is 2,894 counts, second time is 5,547 and third time is 5,557.
- After Downloading only 5,557 images are passed and 1,000 are failed to download, so failed one can be considered as fake. Even though images are downloaded have to check whether any duplicate images are repeated because for model train and analyze their should not be repeated images, then the model will be affected in accuracy. After removing duplicate images, final count is 5114, so, the overall percentage is 78%.

# Data Splitting:
- Before preprocessing data must be split in 70/15/15, it is splitted in stratified way train/test/validation, where the overall combine real and fake counts 5,114 is splitted for train -> 3581 images, val -> 765 images, test -> 768 images keeping the Real/Fake ratio equal in each.

# Data Preprocessing:
- In data preprocessing, image resize concept is used because sizes vary widely: 127 to 1080 px wide and 128 to 2440 px high, so every image has to be resized. According to our project requirement all images are resized as 224 x 224 pixels, convert to a tensor, normalize with the ImageNet mean and standard deviation.
- Augmentation done for training data only: random horizontal flip and random rotation up to 10 degrees. Validation and test images are never augmented.
A custom Dataset loads one image at a time. Dataloader batches them (batch size 32) and shuffles only the training data. No shuffle done in validation and test.

# EDA:
- Exploratory Data Analysis (EDA) is used to summarize, visualize, and understand the dataset.
## Real vs Fake images Before and After:
<img width="1353" height="512" alt="real_vs_fake_before_after" src="https://github.com/user-attachments/assets/ab3eb40a-f794-41ae-9f31-5acce5febef9" />
- Image count, Before cleaning real -> 2,790 (42.5%) and fake -> 3,767 (57.5%) and After cleaning real -> 2,368 (46.3%) and fake -> 2,746 (53.7%). From this we can say more than 1,000 images are fake and some count reduced in real is duplicate.

## Gender Count Before and After:
<img width="1353" height="572" alt="gender_dist_before_after" src="https://github.com/user-attachments/assets/1c791727-74f5-4e57-b847-4ef5b69bf370" />
- Gender count before cleaning Unknown -> 4,507 , Female -> 1,033 & Male -> 1,017,  whereas after cleaning Unknown -> 3,354, Female -> 890 & Male -> 870. From this we can understand moreover counts are affected for both genders.

## Age group count before and after:
<img width="1352" height="570" alt="image" src="https://github.com/user-attachments/assets/8e2fcfe2-bc68-4df7-aec4-07b1f7a6557d" />
- Age counts before cleaning 18-25 -> 1681, 36-50 -> 1670, 26-35 -> 1650 & 50+ -> 1556, whereas after cleaning 18-25 -> 1318, 36-50 -> 1301, 26-35 -> 1258 & 50+ -> 1237.

<img width="608" height="555" alt="age_group_wise_dist_real_and_fake_images" src="https://github.com/user-attachments/assets/bfbacdf3-ddcb-4835-96f8-ced09a94a862" />
- From this picture we're come's to know that real and fake images are somewhat equally placed.








# Model Deployment:

https://deepfakedetectionusingdeeplearning-2y6kp2b7zdvwcgk2nj5hn9.streamlit.app/
