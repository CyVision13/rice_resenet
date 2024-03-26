import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def perform_augmentation(directory, num_augmentations, image_generator,batch_size):
    labels = []
    data = []
    train_data_count = 0  # Counter for data before augmentation

    # Replace backslashes with forward slashes
    directory = directory.replace("\\", "/")

    classes = sorted(os.listdir(directory))
    classes = [c for c in classes if len(c) <= 2]
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

            train_data_count += 1

            # Apply multiple augmentations
            for _ in range(num_augmentations):
                augmented_img = image_generator.random_transform(img_resized)
                data.append(augmented_img)
                labels.append(class_name)

    print("Data loaded before augmentation:")
    print("Train set size:", train_data_count)

    if len(labels) == 0:
        raise ValueError("No labels found. Make sure the directory structure and class names are correct.")

    X = np.array(data)
    y = np.array(labels)

    del data  # Free up memory

    # Normalize inputs by dividing by 255
    X = X / 255.0

    # One-hot encode the labels
    label_map = {class_name: i for i, class_name in enumerate(classes)}
    y = np.array([label_map[label] for label in labels])
    y = to_categorical(y, len(classes))
    print("Number of categories in labels:", y.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    batch_size = batch_size  # Adjust the batch size as needed
    print('batch_size: '+ str(batch_size))
    train_data_generator = image_generator.flow(X_train, y_train, batch_size=batch_size)

    print("Data loaded after augmentation:")
    print("Train set size:", len(y_train))
    print("Test set size:", len(y_test))

    return X_train, X_test, y_train, y_test, train_data_generator
