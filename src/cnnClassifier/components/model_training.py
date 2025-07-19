import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path


tf.config.run_functions_eagerly(True)
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        print(f"Loading base model from: {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        print(f"Base model loaded successfully!")

        # ✅ Force recompile with a fresh optimizer & loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model recompiled with fresh optimizer!")


    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            color_mode="rgb"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Debug info
        print(f"Training samples found: {self.train_generator.samples}")
        print(f"Validation samples found: {self.valid_generator.samples}")
        print(f"Detected class indices: {self.train_generator.class_indices}")

        if self.train_generator.samples == 0 or self.valid_generator.samples == 0:
            raise ValueError(
                f"❌ ERROR: No training/validation images found! "
                f"Please check your training_data folder: {self.config.training_data} "
                f"and ensure it contains subfolders for each class with images inside."
            )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        print(f"Saving model to: {path}")
        model.save(path)
        print(f"Model saved successfully!")

    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        if self.steps_per_epoch == 0 or self.validation_steps == 0:
            raise ValueError(
                f"❌ ERROR: Computed steps are zero! "
                f"Check batch size and number of samples."
            )

        print(f"Starting training for {self.config.params_epochs} epochs...")
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        return history