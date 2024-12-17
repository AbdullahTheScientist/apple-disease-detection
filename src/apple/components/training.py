from pathlib import Path
from apple.entity.config_entity import TrainingConfig
import tensorflow as tf



class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initialize the Training class with a configuration object.
        """
        self.config = config

    def get_base_model(self):
        """
        Load the pre-trained InceptionV3 model as the base model.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
        Create data generators for training and validation with data augmentation (optional).
        """

        # Rescaling images and validation split
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  # Normalize pixel values between 0 and 1
            validation_split=0.20  # 20% data for validation
        )

        # Image generator settings for resizing and batching
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Remove channel info for target size
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation Data Generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training Data Generator with optional augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=30,  # Rotate up to 30 degrees
                horizontal_flip=True,  # Flip images horizontally
                width_shift_range=0.2,  # Shift width by 20%
                height_shift_range=0.2,  # Shift height by 20%
                shear_range=0.2,  # Apply shearing transformations
                zoom_range=0.2,  # Zoom into the image
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,  # Shuffle training data
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.
        """
        model.save(path)

    def train(self, callback_list: list):
        """
        Train the InceptionV3 model using the prepared generators.
        """

        # Calculate steps for training and validation
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
