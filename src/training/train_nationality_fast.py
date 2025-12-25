import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import argparse

def train_nationality_fast(data_dir, epochs=20, batch_size=32):
    print(f"Starting FAST Training on {data_dir}...")
    
    # OPTIMIZATION 1: Reduced Image Size
    # MobileNetV2 works well with 128x128 or even 96x96.
    IMG_SIZE = (128, 128) 
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Directory {train_dir} not found.")
        return

    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = train_generator.num_classes
    
    # OPTIMIZATION 2: MobileNetV2 Architecture
    # much lighter than VGG16 (3.5M params vs 138M params)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # Keep base trainable=False initially, or fine-tune specific layers
    # For speed, we freeze it.
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Better than Flatten for MobileNet
    x = Dense(128, activation='relu')(x) # Smaller dense layer
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        'models/nation_model_best.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    # OPTIMIZATION 3: Aggressive Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    
    # We can also limit steps per epoch for debugging/fast checks if needed, 
    # but with MobileNet + 128x128, it should be fast enough normally.
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )

    model.save('models/nation_model_final.h5')
    print("Fast Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    train_nationality_fast(args.data_dir, args.epochs)
