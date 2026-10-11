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
-  

The source file listed 6,557 image links, with each image’s label (REAL or FAKE), gender and age group. After downloading and cleaning, 5,557 verified images remained. All 1,000 failed downloads were FAKE, so the classes ended up about 50/50. The data was split 70/15/15 into train (3,891), validation (832) and test (834) images, keeping the REAL/FAKE ratio equal in each.

# Model Deployment:

https://deepfakedetectionusingdeeplearning-2y6kp2b7zdvwcgk2nj5hn9.streamlit.app/
