# 🚘 Automatic Number Plate Recognition (ANPR)

<p align="center">
  <strong>AI-Powered Vehicle Number Plate Detection and Recognition System</strong>
</p>

<p align="center">
  Built using Python, YOLO, OpenCV and Tesseract OCR
</p>

---

## 📌 Project Overview

Automatic Number Plate Recognition (ANPR) is a Python-based computer vision project that detects vehicle number plates from images, extracts registration numbers using Optical Character Recognition (OCR), and verifies vehicle authorization through a database.

The system combines YOLO object detection, OpenCV image processing and Tesseract OCR to recognize Indian vehicle registration numbers.

## 🖼️ Project Screenshots

### 1. Input Vehicle Image

The system accepts a vehicle image as input.

![Input Vehicle Image](screenshots/input-image.png)

### 2. Number Plate Detection

YOLO detects the number plate region in the vehicle image.

![Number Plate Detection](screenshots/detection-result.png)

### 3. Cropped Number Plate

The detected number plate is extracted for OCR processing.

![Cropped Number Plate](screenshots/cropped-plate.png)

### 4. OCR Recognition Result

Tesseract OCR extracts the registration characters from the processed number plate.

![OCR Recognition Result](screenshots/ocr-result.png)

### 5. Vehicle Authorization Result

The recognized registration number is checked against the vehicle database to determine its authorization status.

![Vehicle Authorization Result](screenshots/authorization-result.png)

> Replace these example image paths with your actual screenshots. The images will appear on GitHub once the corresponding files are uploaded.

---

## ✨ Key Features

- Vehicle number plate detection using YOLO.
- Number plate extraction using OpenCV.
- Image preprocessing for OCR.
- Character recognition using Tesseract OCR.
- Multiple OCR Page Segmentation Modes (PSM).
- Indian vehicle registration number validation.
- Duplicate detection removal.
- OCR candidate scoring and consensus-based recognition.
- Manual OCR verification.
- Vehicle authorization checking through a CSV database.
- CSV-based recognition and verification reports.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO (Ultralytics) | Number plate detection |
| OpenCV | Image processing |
| Tesseract OCR | Character recognition |
| Pytesseract | Python interface for Tesseract |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Regular Expressions | Number plate validation |
| CSV | Vehicle database and reports |

## ⚙️ System Workflow

```text
Vehicle Image Input
        |
        v
YOLO Number Plate Detection
        |
        v
Number Plate Extraction
        |
        v
OpenCV Image Preprocessing
        |
        v
Tesseract OCR Recognition
        |
        v
Registration Number Validation
        |
        v
OCR Scoring and Consensus
        |
        v
Manual Verification (if required)
        |
        v
Vehicle Database Verification
        |
        v
Final Recognition Result
```

## 📂 Main Project Modules

| File | Description |
|---|---|
| `full_anpr.py` | Main detection and recognition workflow |
| `ocr.py` | Number plate OCR processing |
| `anpr_pipeline.py` | Recognition and authorization processing |
| `database.py` | Vehicle database checking |
| `create_ocr_review.py` | Creates OCR review records |
| `manual_verify_ocr.py` | Manual OCR verification |

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhanushya-A/ANPR_Project.git
cd ANPR_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
pip install ultralytics opencv-python pytesseract pandas numpy
```

### 5. Install Tesseract OCR

Install Tesseract OCR separately on your computer.

Configure its executable path in the Python source code according to your local installation.

## ▶️ Running the Project

Configure the trained YOLO model path and vehicle database path in the source code before execution.

Run the main script using your actual project structure and an input vehicle image.

Example:

```bash
python src/full_anpr.py "path_to_vehicle_image.jpg"
```

If your main script is located directly in the project root, adjust the command accordingly.

## 📊 Output

The system can produce:

- Detected vehicle number plate images.
- Extracted registration numbers.
- OCR confidence and recognition scores.
- Registration number validation details.
- Vehicle authorization status.
- CSV-based recognition reports.

## 🔐 Data Privacy

Vehicle registration records, training datasets, trained model weights and generated output files are maintained locally and are not included in this public repository.

Only use synthetic or authorized sample vehicle records when demonstrating the project.

## 🚀 Future Enhancements

- Real-time number plate recognition using video.
- Improved recognition under different lighting conditions.
- Web-based monitoring dashboard.
- Support for additional vehicle registration formats.
- Performance optimization for real-time processing.

## 👩‍💻 Author

**Dhanushya A**

B.Sc. Information Technology Graduate

- GitHub: [Dhanushya-A](https://github.com/Dhanushya-A)
- LinkedIn: [Dhanushya A](https://www.linkedin.com/in/dhanushya01/)

---

<p align="center">
  <strong>Automatic Number Plate Recognition using AI and Computer Vision</strong>
</p>
