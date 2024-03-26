import os
from import_libraries import *
from data_preprocessing import data_preprocessing
from data_augmentation import perform_augmentation
from model_training import train_model
from plot_and_evaluate import plot_and_evaluate
from install_dependencies import install_dependencies

# Call the install_dependencies function
install_dependencies()

# Define the directory for storing results
results_dir = './../results/TrainData/001'
os.makedirs(results_dir, exist_ok=True)

# Perform data preprocessing
directory = "./../data/datasets/8class_100px"
image_generator, X_train, X_test, y_train, y_test = data_preprocessing(directory)

# Perform augmentation
num_augmentations = 0

batch_size = 4
total_epochs = 12

X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented, train_data_generator = perform_augmentation(directory, num_augmentations, image_generator,batch_size)

# Train the model
model, history = train_model(directory, train_data_generator, X_test_augmented, y_test_augmented, results_dir, total_epochs)

# Plot accuracy and loss and evaluate the model
plot_and_evaluate(history, model, X_test_augmented, y_test_augmented, results_dir, directory)
