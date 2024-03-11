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
results_dir = 'work/TrainData/026/results'
os.makedirs(results_dir, exist_ok=True)

# Perform data preprocessing
directory = "work/datasets/8class_100px"
image_generator, X_train, X_test, y_train, y_test = data_preprocessing(directory)

# Perform augmentation
num_augmentations = 2
batch_size = 128
train_data_generator = image_generator.flow(X_train, y_train, batch_size=batch_size)
X_test_augmented = image_generator.flow(X_test, batch_size=batch_size, shuffle=False)

# Train the model
model, history = train_model(directory, train_data_generator, X_test_augmented, y_test, results_dir)

# Plot accuracy and loss and evaluate the model
plot_and_evaluate(history, model, X_test_augmented, y_test, results_dir)
