# Internship Report: Nationality Detection System

**Task 4 – Nationality Detection Model**

**Author:** Tarrush Saxena  
**Date:** January 2026

---

## Introduction

This report documents the development of a Nationality Detection System as part of my internship assignment. The project required building a machine learning application capable of predicting a person's nationality from their facial image and applying conditional logic to determine additional predictions based on the detected nationality.

The system features a graphical user interface that allows users to upload images or use a live camera feed, with results displayed as visual overlays on the input media.

---

## Background

The task required developing a multi-model computer vision pipeline with the following specifications:
- Detect nationality from facial images
- Apply branching logic based on nationality:
  - **Indian:** Predict emotion, age, and dress color
  - **United States:** Predict emotion and age
  - **African:** Predict emotion and dress color
  - **Other nationalities:** Predict emotion only
- Provide a GUI with image preview and results display

This project sits at the intersection of deep learning, computer vision, and GUI development, requiring knowledge of convolutional neural networks, image processing, and desktop application frameworks.

---

## Learning Objectives

1. Understand and implement multi-task learning pipelines
2. Apply transfer learning using pre-trained models (MobileNetV2)
3. Develop conditional logic systems for attribute prediction
4. Build interactive desktop applications with Tkinter
5. Integrate deep learning models into real-time applications

---

## Activities and Tasks

### Model Development
- Researched suitable datasets: FairFace for nationality, FER2013 for emotions, UTKFace for age
- Implemented transfer learning with MobileNetV2 as the base architecture
- Trained separate models for nationality classification, emotion detection, and age regression
- Developed a color extraction module using K-Means clustering for dress color prediction

### GUI Implementation
- Built a Tkinter-based graphical interface with:
  - Live camera feed functionality
  - Image upload capability with file dialog
  - Real-time bounding box overlays
  - Attribute display panels
  - Execution logging panel

### System Integration
- Created a NationalityEngine class to orchestrate all model predictions
- Implemented branching logic in a configuration file for maintainability
- Integrated all components into a cohesive application

---

## Skills and Competencies

### Technical Skills Developed
- **Deep Learning:** Model training, transfer learning, hyperparameter tuning
- **Computer Vision:** Face detection with Haar cascades, image preprocessing, color analysis
- **Python Development:** Object-oriented programming, modular code architecture
- **GUI Development:** Tkinter widgets, threading for responsive UI, event handling
- **Data Processing:** Dataset preparation, normalization, augmentation

### Soft Skills Enhanced
- Problem decomposition and planning
- Documentation and code organization
- Debugging and troubleshooting complex systems

---

## Feedback and Evidence

The system successfully meets the project requirements:

1. **Nationality Detection:** The model classifies faces into multiple nationality categories
2. **Branching Logic:** Correct implementation verified:
   - Indian faces show: Emotion, Age, Dress Color
   - US faces show: Emotion, Age
   - African faces show: Emotion, Dress Color
   - Other faces show: Emotion only
3. **GUI Features:**
   - Image upload with preview functionality
   - Live camera feed option
   - Bounding boxes with color-coding by nationality
   - Attribute panels displayed alongside detected faces

---

## Challenges and Solutions

### Challenge 1: Model Loading Performance
**Problem:** Initial loading of three models caused the GUI to freeze.  
**Solution:** Implemented threaded model loading with a loading indicator to maintain UI responsiveness.

### Challenge 2: Dress Color Detection Accuracy
**Problem:** Color detection was inconsistent with varying lighting conditions.  
**Solution:** Used K-Means clustering with multiple clusters and selected the dominant cluster, improving robustness.

### Challenge 3: Branching Logic Configuration
**Problem:** Initial implementation had all branches performing all predictions.  
**Solution:** Refactored the configuration file to clearly define prediction sets for each nationality category.

### Challenge 4: Image Upload Integration
**Problem:** Camera feed and image upload functionality needed to coexist.  
**Solution:** Implemented logic to stop the camera when uploading an image, preventing resource conflicts.

---

## Outcomes and Impact

### Deliverables
- Fully functional nationality detection application
- Trained models for nationality, emotion, and age prediction
- Color extraction module for dress color analysis
- Professional GUI with dual input modes
- Comprehensive documentation

### Technical Achievements
- Successfully integrated multiple deep learning models
- Implemented efficient branching logic system
- Created a user-friendly interface for non-technical users
- Achieved real-time processing on live camera feed

---

## Conclusion

This project provided hands-on experience with multi-model machine learning systems and their integration into desktop applications. The conditional branching logic requirement added complexity beyond simple classification, requiring careful system design.

Key takeaways from this project include:
- The importance of modular architecture for maintainability
- How threading improves user experience in GUI applications
- The value of configuration-driven logic for flexibility
- Techniques for integrating multiple ML models efficiently

The completed system demonstrates proficiency in deep learning, computer vision, and software engineering principles, providing a solid foundation for future development in AI applications.

---

*Report submitted as part of internship requirements.*
