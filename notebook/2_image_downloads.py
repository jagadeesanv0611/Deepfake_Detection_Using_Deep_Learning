import requests
import pandas as pd
import os
from tqdm import tqdm

# Step 1: Read the CSv files
deepfake_df = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\projects_of_guvi\Final_project_Deep_learning\data\dataset.csv")

# Step 2: To save images in a folder
os.makedirs(r"C:\Users\jagad\Documents\my_classes\Tasks\projects_of_guvi\Final_project_Deep_learning\data\raw_images", exist_ok=True)

# Step 3: To use all data
sample = deepfake_df

# To track failed URLs 
failed_list = [] 

for index, row in tqdm(sample.iterrows(), total=len(sample)):
    url = row['image_url']
    image_id = row['image_id']
    save_path = f"C:/Users/jagad/Documents/my_classes/Tasks/projects_of_guvi/Final_project_Deep_learning/data/raw_images/{image_id}.jpg"

    # If it has already been downloaded, skip it. Re-running will resume the download
    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        failed_list.append({"image_id": image_id, "url": url, "error": str(e)})

print(f"\nTotal downloaded: {len(sample) - len(failed_list)}")
print(f"Total failed: {len(failed_list)}")

# Save the failed list as a CSV for review.
if failed_list:
    pd.DataFrame(failed_list).to_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\projects_of_guvi\Final_project_Deep_learning\data\failed_downloads.csv", index=False)
    print("Failed URLs saved to data/failed_downloads.csv")



