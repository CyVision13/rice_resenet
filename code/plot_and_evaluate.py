import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_and_evaluate(history, model, X_test, y_test, results_dir, directory):

    classes = sorted(os.listdir(directory))
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

    # Save the figure
    save_path_loss = os.path.join(results_dir, 'loss_plot.png')
    plt.savefig(save_path_loss)
    print('Loss plot saved successfully at:', save_path_loss)
    plt.show()

    # Plotting train and validation loss, and train and validation accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Get the training and validation loss values from the history object
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the training and validation loss values
    ax1.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    ax1.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Get the training and validation accuracy values from the history object
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Plot the training and validation accuracy values
    ax2.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    ax2.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot as an image
    save_path_accuracy = os.path.join(results_dir, 'loss_and_accuracy_plot.png')
    plt.savefig(save_path_accuracy)
    print('Loss and accuracy plot saved successfully at:', save_path_accuracy)

    # Show the plot
    plt.show()

    # Evaluate the model on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Create the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.show()
