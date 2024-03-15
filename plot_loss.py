import os
import matplotlib.pyplot as plt

def plot_loss(history, results_dir):
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
    save_path = os.path.join(results_dir, 'loss_plot.png')
    plt.savefig(save_path)

    print('Image saved successfully at:', save_path)
