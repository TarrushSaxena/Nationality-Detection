import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Conv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import argparse
import numpy as np

def train_emotion_fast(data_dir, epochs=30, batch_size=32):
    print(f"Starting FAST Training (MobileNetV2) for Emotion on {data_dir}...")
    
    # Image Size: MobileNetV2 minimal size is 32x32, we use 48x48 (FER size) -> Resized to 96x96 for better feature extraction
    IMG_SIZE = (96, 96) 
    
    # 1. Data Generators
    # Note: FER is grayscale, but MobileNet requires 3 channels. 
    # specific logic needed: check if we can force flow_from_directory to load as RGB (it copies channels)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # If using single train dir
    )

    # Determine if 'test' folder exists or we just split 'train'
    if os.path.exists(os.path.join(os.path.dirname(data_dir), 'test')):
        val_dir = os.path.join(os.path.dirname(data_dir), 'test')
        subset = None
    else:
        val_dir = data_dir
        subset = 'validation'

    # TRAIN
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        color_mode='rgb', # Force 3 channels even if grayscale source
        batch_size=batch_size,
        class_mode='categorical',
        subset='training' if subset else None
    )

    # VAL
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation' if subset else None
    )

    num_classes = train_generator.num_classes
    
    # 2. Model: MobileNetV2 (Transfer Learning)
    # Weights='imagenet' gives "Advanced AI" head start
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(96, 96, 3)
    )
    
    # Freeze base model for speed and stability initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        'models/emotion_model_best.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # 4. Train
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    model.save('models/emotion_model_final.h5')
    print("Fast Emotion Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to FER-2013 train folder")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    train_emotion_fast(args.data_dir, args.epochs)
