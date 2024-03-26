import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

 ###
print('Loading Model')
# Load the saved model
model_path = 'work/TrainData/026/best_model.h5'
model = load_model(model_path)

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate and print the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# Create a directory to save the results
results_dir = 'work/TrainData/026/results'
os.makedirs(results_dir, exist_ok=True)

# Save the confusion matrix
np.save(os.path.join(results_dir, 'confusion_matrix.npy'), cm)

# Save the predicted and true classes
np.save(os.path.join(results_dir, 'predicted_classes.npy'), y_pred_classes)
np.save(os.path.join(results_dir, 'true_classes.npy'), y_true_classes)

print("Results saved in the 'results' folder.")

# Plotting train loss and test loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'r', label='Train Loss')
plt.plot(epochs, test_loss, 'b', label='Test Loss')

plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the figure
save_dir = 'work/TrainData/026/results'  # Replace with your desired directory path
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
save_path = os.path.join(save_dir, 'loss_plot.png')  # Replace 'loss_plot.png' with your desired filename
plt.savefig(save_path)

print('Image saved successfully at:', save_path)
