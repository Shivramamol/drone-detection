# 🚁 AI Drone Detection System

## Overview

The AI Drone Detection System is a computer vision application developed using Python, Streamlit, and YOLOv8. The system detects and classifies objects as either **Drone** or **Bird** from uploaded images and videos.

The project uses a custom-trained YOLOv8 model trained on a labeled dataset of drone and bird images. The application provides an interactive web interface where users can upload media files, view detection results, analyze confidence scores, and download processed outputs.

---

## Features

### Image Detection

* Upload JPG, JPEG, or PNG images
* Detect drones and birds
* Display bounding boxes around detected objects
* Show confidence scores
* Download processed image

### Video Detection

* Upload MP4, AVI, or MOV videos
* Process video frame-by-frame
* Detect drones and birds throughout the video
* Generate processed output video

### Detection Analytics

* Total objects detected
* Number of drones detected
* Number of birds detected
* Detection confidence table

### User Interface

* Built using Streamlit
* Interactive confidence threshold slider
* Side-by-side comparison of original and detected images
* Professional dashboard layout

---

## Technologies Used

* Python
* Streamlit
* YOLOv8 (Ultralytics)
* OpenCV
* Pillow (PIL)
* Pandas
* Google Colab
* Deep Learning
* Computer Vision

---

## Project Architecture

User Uploads Image/Video

↓

Streamlit Frontend

↓

YOLOv8 Custom Model (best.pt)

↓

Object Detection

↓

Bounding Boxes + Confidence Scores

↓

Detection Statistics

↓

Result Display & Download

---

## Dataset

The model was trained using a custom object detection dataset containing two classes:

| Class ID | Class Name |
| -------- | ---------- |
| 0        | Bird       |
| 1        | Drone      |

### Dataset Preparation

1. Image collection
2. Image annotation using Roboflow
3. Dataset splitting

   * Train: 70%
   * Validation: 20%
   * Test: 10%
4. Export in YOLOv8 format

---

## Model Training

### Training Environment

* Google Colab
* NVIDIA Tesla T4 GPU

### Model

YOLOv8 Nano (yolov8n.pt)

### Training Parameters

| Parameter  | Value              |
| ---------- | ------------------ |
| Epochs     | 50                 |
| Image Size | 640                |
| Classes    | 2                  |
| Framework  | Ultralytics YOLOv8 |

---

## Project Structure

```text
DroneDetection/
│
├── app.py
├── best.pt
│
├── pages/
│   ├── 1_Image_Detection.py
│   ├── 2_Video_Detection.py
│   └── 3_About_Project.py
│
├── uploads/
├── outputs/
│
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd DroneDetection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit ultralytics opencv-python pillow pandas
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Usage

### Image Detection

1. Open the application.
2. Navigate to Image Detection.
3. Upload an image.
4. Adjust confidence threshold if required.
5. Click "Detect Objects".
6. View results and download the processed image.

### Video Detection

1. Navigate to Video Detection.
2. Upload a video file.
3. Click "Process Video".
4. Wait for detection to complete.
5. Download or view the processed video.

---

## Results

The system is capable of:

* Detecting drones in images
* Detecting birds in images
* Displaying confidence scores
* Generating annotated output media
* Providing object count statistics

---

## Future Enhancements

* Real-time webcam detection
* CCTV surveillance integration
* Drone tracking and trajectory prediction
* Alert and notification system
* Cloud deployment
* Mobile application support
* Multi-drone detection and tracking
* Night-time drone detection

---

## Learning Outcomes

Through this project, the following concepts were implemented:

* Object Detection
* Deep Learning
* Computer Vision
* Dataset Annotation
* Model Training
* YOLOv8 Framework
* Streamlit Application Development
* AI Model Deployment

---

## Developer

**Shivram Parkhi**

Diploma Student | AI & Computer Vision Enthusiast

---

## License

This project is developed for educational and research purposes.
