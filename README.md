# Face Recognition System

## Description

The **Face Recognition System** utilizes the **face_recognition** library, powered by deep learning models, to effectively detect and recognize faces. This system comprises several stages: data loading, model training, model evaluation, and real-time face recognition, including a mechanism for spoof detection to enhance security.


## Features

- **Data Loading**: Organizes student images into structured folders for training, testing, and validation. Each student's images are stored in unique sub-folders.
  
- **Model Training**: Loads face encodings and student IDs, iterating through multiple epochs to compare encodings, calculate predictions, and select the best model based on validation accuracy.

- **Model Evaluation**: Assesses performance using a separate test dataset and calculates metrics like accuracy and AUC (Area Under the Curve).

- **Spoof Detection**: Incorporates a liveness detection mechanism that analyzes image texture to differentiate between genuine faces and potential spoof attempts.

- **Real-Time Recognition**: Captures video frames, detects faces, and allows for user interaction when unknown faces are encountered.

- **User Interface**: Built with **tkinter** for dialogs and **matplotlib** for real-time video feed, featuring buttons for marking attendance and quitting the application.


## Results

The model achieved impressive validation and test accuracies, but results suggest potential overfitting due to the small dataset size. The spoof detection algorithm proved effective, minimizing false positives.


## Future Enhancements

- **Advanced Liveness Detection**: Implement machine learning techniques for more sophisticated spoof detection.
  
- **Scalability**: Enhance system capacity to accommodate larger datasets.

- **User Experience**: Refine the UI for better interaction and response times.


## Getting Started

1. **Installation**: Ensure you have the **face_recognition** library installed, along with its dependencies.
2. **Dataset Preparation**: Organize images in the specified folder structure for known students.
3. **Run the System**: Execute the main script to start the face recognition and attendance marking process.

For more detailed instructions on setting up the environment and using the system, please refer to the provided documentation and examples.
