# 🚘 Automatic Number Plate Recognition (ANPR)

**AI-Powered Vehicle Number Plate Detection and Recognition System**

Built using Python, YOLO, OpenCV, and Tesseract OCR.

---

## 📌 Project Overview

Automatic Number Plate Recognition (ANPR) is a Python-based computer vision project designed to detect vehicle number plates from images, recognize registration numbers using Optical Character Recognition (OCR), and verify vehicle authorization through a database.

The system combines YOLO object detection, OpenCV image processing, and Tesseract OCR to recognize Indian vehicle registration numbers.

## ✨ Key Features

- Vehicle number plate detection using YOLO.
- Number plate extraction from vehicle images.
- Image preprocessing using OpenCV.
- Character recognition using Tesseract OCR.
- Multiple OCR configurations for improved recognition.
- Indian vehicle registration number validation.
- Duplicate detection removal.
- OCR candidate scoring and consensus-based recognition.
- Manual OCR verification.
- Vehicle authorization checking using a CSV database.
- CSV-based recognition and verification reports.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO (Ultralytics) | Number plate detection |
| OpenCV | Image processing |
| Tesseract OCR | Text recognition |
| Pytesseract | Python interface for Tesseract |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Regular Expressions | Registration number validation |
| CSV | Vehicle database and reports |

## ⚙️ System Architecture

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

## 📂 Project Modules

| Module | Description |
|---|---|
| `full_anpr.py` | Main number plate detection and recognition workflow |
| `ocr.py` | OCR processing and text recognition |
| `anpr_pipeline.py` | Recognition and vehicle authorization processing |
| `database.py` | Vehicle database verification |
| `create_ocr_review.py` | Preparation of OCR results for manual review |
| `manual_verify_ocr.py` | Manual OCR verification |

## 🔍 How It Works

1. **Image Input:** Accepts a vehicle image for processing.
2. **Number Plate Detection:** Uses YOLO to locate number plates.
3. **Plate Extraction:** Extracts detected number plate regions.
4. **Image Preprocessing:** Uses OpenCV to prepare images for OCR.
5. **Text Recognition:** Uses Tesseract OCR to extract registration characters.
6. **Validation:** Checks recognized text against Indian registration number patterns.
7. **OCR Consensus:** Evaluates multiple OCR candidates to improve recognition reliability.
8. **Manual Verification:** Allows uncertain recognition results to be reviewed.
9. **Database Verification:** Checks whether a vehicle is authorized and active.
10. **Result Generation:** Saves recognition and authorization results.

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Dhanushya-A/ANPR_Project.git
cd ANPR_Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install ultralytics opencv-python pytesseract pandas numpy
```

### 5. Install Tesseract OCR

Install Tesseract OCR separately on your system.

Configure the Tesseract executable path in the Python source code according to your installation.

## ▶️ Execution

Before running the application:

- Configure the trained YOLO model path.
- Configure the vehicle database path.
- Install Tesseract OCR.
- Update local file paths in the source code.

Run the main ANPR script with a vehicle image:

```bash
python src/full_anpr.py "path_to_vehicle_image.jpg"
```

Adjust the command according to the actual location of your main Python script.

## 📊 Output

The system can generate:

- Detected number plate images.
- Recognized vehicle registration numbers.
- OCR confidence and recognition scores.
- Registration number validation details.
- Vehicle authorization status.
- CSV-based recognition reports.

## 🔐 Data Privacy

Training datasets, vehicle registration records, trained model weights, and generated output files are maintained locally and are not included in this public repository.

Only use synthetic or authorized vehicle records when demonstrating the project.

## 🚀 Future Enhancements

- Real-time number plate recognition using video streams.
- Improved OCR accuracy under different lighting conditions.
- Web-based vehicle monitoring dashboard.
- Support for additional vehicle registration formats.
- Performance optimization for real-time applications.

## 👩‍💻 Author

**Dhanushya A**

B.Sc. Information Technology Graduate

- GitHub: https://github.com/Dhanushya-A
- LinkedIn: https://www.linkedin.com/in/dhanushya01/

---

**Automatic Number Plate Recognition using Artificial Intelligence and Computer Vision.**
