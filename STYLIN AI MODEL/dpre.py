import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
style_df = pd.read_csv('d:/archive/styles.csv', error_bad_lines=False)

# Data cleaning
style_df.dropna(subset=['id', 'subCategory'], inplace=True)
style_df['id'] = style_df['id'].astype(str)

# Load and preprocess images
image_folder_path = 'd:/archive/images/'

def process_images_in_batches(batch_size):
    batch_images = []
    batch_labels = []
    
    for _, row in style_df.iterrows():
        if len(batch_images) == batch_size:
            yield np.array(batch_images, dtype='float32'), np.array(batch_labels)
            batch_images, batch_labels = [], []

        image_path = os.path.join(image_folder_path, row['id'] + '.jpg')
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                img = img.resize((224, 224)).convert('RGB')
                img_array = np.array(img, dtype='float32') / 255.0
                batch_images.append(img_array)
                batch_labels.append(row['subCategory'])

    if batch_images:
        yield np.array(batch_images, dtype='float32'), np.array(batch_labels)

# Example usage:
for X_batch, y_batch in process_images_in_batches(batch_size=100):
    # Split the batch dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size=0.2, random_state=42)
    # Do something with X_train, X_test, y_train, y_test, like fitting the model
   # print(X_train.shape, y_train.shape)
    # Note: You would typically feed this batch directly into your training process
