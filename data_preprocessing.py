import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
def data_preprocessing(directory):
    classes = sorted(os.listdir(directory))
    classes = [c for c in classes if len(c) <= 2]
    print("Number of classes:", len(classes))

    data = []
    labels = []

    for class_name in classes:
        path = os.path.join(directory, class_name)
        if not os.path.isdir(path):
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_resized = cv2.resize(img, (100, 100))
            data.append(img_resized)
            labels.append(class_name)

    # Convert lists to numpy arrays
    X = np.array(data)
    y = np.array(labels)

    # Normalize inputs by dividing by 255
    X = X / 255.0

    # One-hot encode the labels
    label_map = {class_name: i for i, class_name in enumerate(classes)}
    y = np.array([label_map[label] for label in labels])

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameters for data augmentation
    augmentation_params = {
        'rotation_range': 30,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }

    # Create an ImageDataGenerator object for data augmentation
    image_generator = ImageDataGenerator(**augmentation_params)

    return image_generator, X_train, X_test, y_train, y_test
