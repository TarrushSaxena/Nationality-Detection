# Nationality Detection System

A multi-task computer vision system that predicts nationality, emotion, age, and dress color from facial images using deep learning with conditional branching logic.

## Overview

This project implements a machine learning pipeline that:
1. Detects faces in images or video streams
2. Predicts the nationality of each detected face
3. Applies nationality-specific branching logic to determine additional predictions

## Features

- **Nationality Detection** – Classifies ethnicity using a MobileNetV2-based model trained on FairFace
- **Emotion Detection** – Recognizes 7 emotions: Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral
- **Age Prediction** – Regression model trained on UTKFace dataset
- **Dress Color Detection** – Extracts dominant clothing color using K-Means clustering
- **Graphical User Interface** – Tkinter-based GUI with live camera feed and image upload support

## Branching Logic

The system applies different attribute predictions based on detected nationality:

| Nationality   | Predictions                      |
|---------------|----------------------------------|
| Indian        | Emotion, Age, Dress Color        |
| United States | Emotion, Age                     |
| African       | Emotion, Dress Color             |
| Other         | Emotion                          |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application
```bash
python main.py
```

### GUI Controls
- **Run System** – Start live camera feed with real-time analysis
- **Upload Image** – Select an image file for analysis (stops camera if running)
- **Close** – Exit the application

The results are displayed as overlays on the image/video with bounding boxes and attribute panels.

## Training Models

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
├── config/           # Configuration files (branching logic, model paths)
├── data/             # Training datasets (not included in repo)
├── models/           # Trained model weights (.h5 files)
├── src/
│   ├── gui/          # Tkinter GUI implementation
│   ├── modules/      # Core logic (NationalityEngine, ColorExtractor)
│   └── training/     # Training scripts for all models
├── tests/            # Unit tests
├── main.py           # Application entry point
└── requirements.txt  # Python dependencies
```

## Requirements

- Python 3.10+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- Pillow

## Author

Tarrush Saxena

## License

This project was developed as part of an internship assignment.
