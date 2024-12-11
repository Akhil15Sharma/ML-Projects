import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split


data_dir = 'D:/Machine_Learning/Brain_Tumor/Training'
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']
image_size = 224

def load_images(data_dir, categories):
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)

        for img in tqdm(os.listdir(path), desc=f'Loading images from {path}'):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_img = cv2.resize(img_array, (image_size, image_size))
                data.append(resized_img)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")

    data = np.array(data)
    labels = np.array(labels)
    return data, labels


X, y = load_images(data_dir, categories)

X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f"Training data shape: {X_train.shape}, Training labels: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels: {y_test.shape}")
