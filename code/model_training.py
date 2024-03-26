import os
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from data_augmentation import perform_augmentation

def train_model(directory, train_data_generator, X_test, y_test, results_dir, total_epochs):
    classes = sorted(os.listdir(directory))
    classes = [c for c in classes if len(c) <= 2]
    activation = 'relu'
    input_shape = (100, 100, 3)
    num_classes = len(classes)
    epochs = total_epochs  # Total number of epochs

    use_pretrained_model = True  # Set this to True to use pre-trained ResNet-50, False for custom model

    if use_pretrained_model:
        # Load the pre-trained ResNet-50 model without the top (classification) layer
        base_model = ResNet50(weights='./../data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)
    else:
        # Import your custom ResNet model
        # base_model = CustomResNet(input_shape=input_shape, num_classes=num_classes)
        pass

    # Create the new classification layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Adjust the learning rate dynamically
    def schedule(epoch, lr):
        if epoch < total_epochs / 3:
            return lr
        elif total_epochs / 3 <= epoch < 2 * total_epochs / 3:
            return 0.0001
        else:
            return 0.00001

    lr_scheduler = LearningRateScheduler(schedule)

    # Adjust the learning rate
    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    checkpoint = ModelCheckpoint(os.path.join(results_dir, 'best_model.h5'), monitor='val_loss', save_best_only=True)

    # Train the model using train_generator
    history = model.fit(
        train_data_generator,
        steps_per_epoch=len(train_data_generator),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )

    return model, history
