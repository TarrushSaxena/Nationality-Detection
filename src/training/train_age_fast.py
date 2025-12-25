import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

class UTKFaceGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=32, dim=(96, 96)):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.dim = dim

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size))

        for i, path in enumerate(batch_paths):
            try:
                # Format: [age]_[gender]_[race]_[date].jpg
                filename = os.path.basename(path)
                age = int(filename.split('_')[0])
                
                img = cv2.imread(path)
                img = cv2.resize(img, self.dim) # Resize to 96x96
                img = img / 255.0 # Normalize
                
                X[i,] = img
                y[i] = age / 116.0 # Normalize output
            except Exception:
                X[i,] = np.zeros((*self.dim, 3))
                y[i] = 0.5 

        return X, y

def train_age_fast(data_dir, epochs=30):
    print(f"Starting FAST Training (MobileNetV2) for Age on {data_dir}...")
    
    # 1. Prepare Data
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    if not files:
        print("No images found.")
        return

    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Generators (Resize to 96x96)
    train_gen = UTKFaceGenerator(train_files, batch_size=32, dim=(96, 96))
    val_gen = UTKFaceGenerator(val_files, batch_size=32, dim=(96, 96))
    
    # 2. Model: MobileNetV2 (Transfer Learning)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(96, 96, 3)
    )
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='linear')(x) # Regression

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    checkpoint = ModelCheckpoint('models/age_model_best.h5', monitor='val_mae', save_best_only=True, mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_mae', patience=5, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
    model.save('models/age_model_final.h5')
    print("Fast Age Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to UTKFace folder")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    train_age_fast(args.data_dir, args.epochs)
