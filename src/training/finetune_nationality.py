"""
Fine-Tuning Script for Nationality Model
This unfreezes the top layers of MobileNetV2 for higher accuracy.
Run AFTER initial training is complete.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import argparse

def finetune_nationality(data_dir, epochs=10, batch_size=32):
    print("Starting Fine-Tuning for Nationality Model...")
    
    IMG_SIZE = (128, 128)
    
    # Load existing best model
    model_path = 'models/nation_model_best.h5'
    if not os.path.exists(model_path):
        print("ERROR: No base model found. Train the model first using train_nationality_fast.py")
        return
    
    model = load_model(model_path, compile=False)
    
    # Unfreeze top 20 layers of MobileNetV2 for fine-tuning
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001), # 10x lower than initial
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Unfroze top 30 layers for fine-tuning.")
    
    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/nation_model_finetuned.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    
    # Train
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print("Fine-Tuning Complete! New model saved to: models/nation_model_finetuned.h5")
    print("To use it, update config/nation_config.py to point to 'nation_model_finetuned.h5'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ethnicity_samples folder")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    finetune_nationality(args.data_dir, args.epochs)
