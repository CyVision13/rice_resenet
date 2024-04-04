# Rice ResNet Image Classification

## Description
This project aims to classify images of rice grains using the Rice ResNet architecture for deep learning-based image classification. It has been developed and tested on CodeOcean, a cloud-based computational research platform.

For replicating the experiments or running the code, you can access the CodeOcean capsule through the following DOI link: [10.24433/CO.7571202.v1](https://doi.org/10.24433/CO.7571202.v1).


## Related Article
This project is related to the research article titled "Deep learning-based image classification of rice grains using Rice ResNet" published in Expert Systems with Applications. You can access the article through the following DOI link: [10.1016/j.eswa.2024.123276](https://doi.org/10.1016/j.eswa.2024.123276).


For inquiries related to the article, you can contact the authors:
- Mohammad Razavi: [m.r.razavi@outlook.com](mailto:m.r.razavi@outlook.com)
- Samira Mavaddati: [s.mavaddati@umz.ac.ir](mailto:s.mavaddati@umz.ac.ir)
- Ziad Kobti: [kobti@uwindsor.ca](mailto:kobti@uwindsor.ca)
- Hamidreza Koohi: [koohi@shomal.ac.ir](mailto:koohi@uwindsor.ca)

## Project Structure
- `.idea/`: PyCharm project settings directory.
- `Lib/`: Libraries directory.
- `models/`: Model files directory.
- `Scripts/`: Scripts directory.
- `share/`: Shared resources directory.
- `work/`: Working directory containing data and training results.
- `__pycache__/`: Cached Python files directory.
- `.gitignore`: Git ignore file.
- `data_augmentation.py`: Script for performing data augmentation.
- `data_preprocessing.py`: Script for data preprocessing.
- `import_libraries.py`: Script for importing necessary libraries.
- `install_dependencies.py`: Script for installing project dependencies.
- `main.py`: Main script for running the project.
- `model_evaluation.py`: Script for evaluating the trained model.
- `model_training.py`: Script for training the model.
- `plot_accuracy.py`: Script for plotting accuracy.
- `plot_and_evaluate.py`: Script for plotting and evaluating the model.
- `plot_loss.py`: Script for plotting loss.
- `pyvenv.cfg`: Python virtual environment configuration file.

## Usage
1. Run `install_dependencies.py` to install project dependencies.
2. Execute `main.py` to preprocess data, perform augmentation, train the model, and evaluate it.

## Customization
- **Data Preprocessing**: The `data_preprocessing` function in `data_preprocessing.py` preprocesses the data by resizing images, normalizing inputs, and performing train-test split.
- **Data Augmentation**: You can customize data augmentation settings in the `perform_augmentation` function in `data_augmentation.py`. Adjust the `num_augmentations` parameter to change the level of augmentation.
- **Epoch Size and Batch Size**: The total number of epochs and batch size can be adjusted in the `main.py` script. Modify the `total_epochs` and `batch_size` variables to change these parameters.
- **Sample Data**: This project uses a small dataset with just 40 images of size 100x100 pixels for each class. Feel free to replace this with your own dataset for testing purposes.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or create a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
