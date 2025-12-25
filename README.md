# Nationality Detection System

A multi-task computer vision system that detects nationality, emotion, age, and dress color using deep learning.

## Features
- **Nationality Detection** - Predicts ethnicity using MobileNetV2 trained on FairFace
- **Emotion Detection** - 7 emotions (Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral)
- **Age Prediction** - Regression model trained on UTKFace
- **Dress Color Detection** - K-Means clustering on torso region
- **Branching Logic** - Different attributes shown based on nationality

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application
```bash
python main.py
```

### Train Models (Optional)
```bash
# Download datasets first
python src/training/download_data.py
python src/training/setup_data.py

# Train models
python src/training/train_nationality_fast.py --data_dir "data/ethnicity_samples" --epochs 10
python src/training/train_emotion_fast.py --data_dir "data/emotion/train" --epochs 30
python src/training/train_age_fast.py --data_dir "data/age" --epochs 30
```

## Project Structure
```
├── config/           # Configuration files
├── data/             # Training datasets (not in repo)
├── models/           # Trained model weights (not in repo)
├── src/
│   ├── gui/          # Tkinter GUI
│   ├── modules/      # Core logic (engine, color extractor)
│   └── training/     # Training scripts
├── tests/            # Unit tests
├── main.py           # Entry point
└── requirements.txt  # Dependencies
```

## Requirements
- Python 3.10+
- TensorFlow 2.x
- OpenCV
- scikit-learn
