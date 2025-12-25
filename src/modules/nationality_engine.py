import cv2
import numpy as np
import os
import sys

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input

# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.nation_config import *
from src.modules.color_extractor import ColorExtractor

class NationalityEngine:
    def __init__(self):
        # Initialize sub-modules
        self.color_extractor = ColorExtractor(k=COLOR_K)
        
        # Load Models (with compile=False to avoid metric errors)
        self.nation_model = self._load_model("nationality")
        self.emotion_model = self._load_model("emotion")
        self.age_model = self._load_model("age")
        
        # Get input sizes from loaded models (auto-detect)
        self.nation_size = self._get_input_size(self.nation_model, (128, 128))
        self.emotion_size = self._get_input_size(self.emotion_model, (96, 96))
        self.age_size = self._get_input_size(self.age_model, (96, 96))
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Labels for classification mapping
        self.nation_labels = [INDIAN, US, AFRICAN, OTHER, "East Asian", "Latino", "Middle Eastern"]
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def _load_model(self, model_type):
        path = MODEL_PATHS.get(model_type, f"models/{model_type}.h5")
        if os.path.exists(path):
            print(f"  [OK] Loading {model_type} model...")
            try:
                # compile=False avoids Keras metric loading errors
                model = load_model(path, compile=False)
                return model
            except Exception as e:
                print(f"  [WARN] {model_type} load failed: {e}")
        
        print(f"  [SKIP] {model_type} model not found, using fallback.")
        return None
    
    def _get_input_size(self, model, default):
        if model is not None:
            try:
                shape = model.input_shape
                if shape and len(shape) >= 3:
                    return (shape[1], shape[2])
            except:
                pass
        return default

    def predict_nationality(self, face_img):
        if self.nation_model is None:
            return "Unknown", 0.0
        
        img = cv2.resize(face_img, self.nation_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        preds = self.nation_model.predict(img, verbose=0)
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        label = self.nation_labels[class_idx % len(self.nation_labels)]
        
        return label, confidence

    def predict_emotion(self, face_img):
        if self.emotion_model is None:
            return "Unknown"
        
        img = cv2.resize(face_img, self.emotion_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        preds = self.emotion_model.predict(img, verbose=0)
        label = self.emotion_labels[np.argmax(preds) % len(self.emotion_labels)]
        return label

    def predict_age(self, face_img):
        if self.age_model is None:
            return 25 # Default if model missing
        
        img = cv2.resize(face_img, self.age_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        age_norm = self.age_model.predict(img, verbose=0)[0][0]
        # Training normalized age by 116, so denormalize
        age = int(age_norm * 116)
        return max(1, min(age, 100)) # Clamp to reasonable range

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Step 1: Nationality
            nation, conf = self.predict_nationality(face_roi)
            
            # Identify Branch
            branch_key = OTHER
            if "ndian" in nation:
                branch_key = INDIAN
            elif "hite" in nation or "US" in nation:
                branch_key = US
            elif "lack" in nation or "frican" in nation:
                branch_key = AFRICAN
            
            # Step 2: Branching Logic
            required_tasks = BRANCHES.get(branch_key, BRANCHES[OTHER])
            
            analysis = {
                "bbox": (x, y, w, h),
                "nationality": nation,
                "confidence": conf,
                "branch": branch_key,
                "attributes": {}
            }
            
            if "Emotion" in required_tasks:
                analysis["attributes"]["Emotion"] = self.predict_emotion(face_roi)
                
            if "Age" in required_tasks:
                analysis["attributes"]["Age"] = self.predict_age(face_roi)
                
            if "Dress Color" in required_tasks:
                color = self.color_extractor.extract_dress_color(frame, (x, y, w, h))
                analysis["attributes"]["Dress Color"] = color
                
            results.append(analysis)
            
        return results
