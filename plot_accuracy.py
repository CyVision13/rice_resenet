import matplotlib.pyplot as plt

def plot_accuracy(history, results_dir):
    # Get the training and validation accuracy values from the history object
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Create a figure with two subplots: and accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the training and validation accuracy values
    ax1.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    ax1.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot as an imagcm
    plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))

    # Show the plot
    plt.show()
