import os
import re
import sys
import cv2
import csv
import numpy as np
import pytesseract
from ultralytics import YOLO


# ================================================================
# PATHS
# ================================================================

MODEL_PATH = r"D:\ANPR_Project\runs\plate_detector_20epochs-2\weights\best.pt"

DATABASE_PATH = r"D:\ANPR_Project\data\vehicles.csv"

OUTPUT_DIR = r"D:\ANPR_Project\outputs\full_test"

PLATE_DIR = os.path.join(OUTPUT_DIR, "plates")


# ================================================================
# SETTINGS
# ================================================================

IMAGE_SIZE = 416

CONF_THRESHOLD = 0.20

# Duplicate removal
IOU_THRESHOLD = 0.30
CONTAINMENT_THRESHOLD = 0.65
CENTER_DISTANCE_RATIO = 0.50


# ================================================================
# TESSERACT
# ================================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ================================================================
# INDIAN STATE / UT CODES
# ================================================================

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL",
    "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
    "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB"
}


# ================================================================
# OCR CORRECTIONS
# Conservative corrections only
# ================================================================

OCR_CORRECTIONS = {
    "ETS": "TS",
    "EITS": "TS",
    "ITS": "TS",
    "T5": "TS",

    "MHO": "MH",
    "MHO3": "MH03",
    "MHO4": "MH04",
    "MHO5": "MH05",

    "DLO": "DL0",
    "KLO": "KL0",
}


# ================================================================
# BASIC CLEANING
# ================================================================

def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# ================================================================
# APPLY CONSERVATIVE OCR CORRECTIONS
# ================================================================

def correct_common_ocr_errors(text):

    text = clean_text(text)

    if not text:
        return text

    # ------------------------------------------------------------
    # Prefix corrections
    # ------------------------------------------------------------

    for wrong, right in OCR_CORRECTIONS.items():

        if text.startswith(wrong):
            text = right + text[len(wrong):]
            break

    if re.match(
        r"^[A-Z]{2}\d[A-Z]{1,3}\d{1,4}$",
        text
    ):

        state = text[:2]

        if state in STATE_CODES:

            candidate = (
                text[:2]
                + "0"
                + text[2:]
            )

            if is_valid_plate_structure(candidate):

                text = candidate

    return text


# ================================================================
# INDIAN PLATE STRUCTURE
# ================================================================

def plate_structure_score(text):

    text = clean_text(text)

    if not text:
        return 0

    # Standard Indian format:
    # XX 00 XX 0000
    #
    # Examples:
    # TN72BB4165
    # MH04DW7092
    # AP10AP9486

    pattern = re.match(
        r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$",
        text
    )

    if pattern:

        state = pattern.group(1)
        number = pattern.group(2)
        letters = pattern.group(3)
        serial = pattern.group(4)

        score = 0

        if state in STATE_CODES:
            score += 4

        if 1 <= len(number) <= 2:
            score += 2

        if 1 <= len(letters) <= 3:
            score += 2

        if 1 <= len(serial) <= 4:
            score += 2

        if len(text) >= 8:
            score += 2

        return score

    # BH series
    bh_pattern = re.match(
        r"^(\d{2})BH(\d{4})([A-Z]{1,2})$",
        text
    )

    if bh_pattern:
        return 12

    return 0


# ================================================================
# STRICT VALIDATION
# ================================================================

def is_valid_plate_structure(text):

    text = clean_text(text)

    if re.match(
        r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$",
        text
    ):

        state = text[:2]

        if state in STATE_CODES:
            return True

    if re.match(
        r"^\d{2}BH\d{4}[A-Z]{1,2}$",
        text
    ):
        return True

    return False


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def create_variants(image):

    variants = []

    # Original
    variants.append(("original", image))

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))

    # Enlarged
    enlarged = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(("enlarged", enlarged))

    # Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(("blurred", blurred))

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    variants.append(("clahe", enhanced))

    # OTSU
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(("otsu", otsu))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants.append(("adaptive", adaptive))

    # Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    variants.append(("sharpened", sharpened))

    # Morphological close
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    morph = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_CLOSE,
        morph_kernel
    )

    variants.append(("morph", morph))

    return variants


# ================================================================
# TESSERACT CONFIDENCE
# ================================================================

def get_tesseract_confidence(image, config):

    try:

        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        confidences = []

        for i, txt in enumerate(data["text"]):

            txt = txt.strip()

            if txt:

                try:
                    conf = float(data["conf"][i])

                    if conf >= 0:
                        confidences.append(conf)

                except:
                    pass

        if confidences:
            return float(np.mean(confidences))

    except:
        pass

    return 0.0


# ================================================================
# OCR
# ================================================================

def run_ocr(image):

    variants = create_variants(image)

    candidates = {}

    psms = [6, 7, 8, 13]

    for variant_name, variant in variants:

        for psm in psms:

            config = (
                f"--oem 3 --psm {psm} "
                f"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            try:

                text = pytesseract.image_to_string(
                    variant,
                    config=config
                )

            except Exception:
                continue

            text = clean_text(text)

            if not text:
                continue

            text = correct_common_ocr_errors(text)

            confidence = get_tesseract_confidence(
                variant,
                config
            )

            structure = plate_structure_score(text)

            # Ignore extremely long OCR garbage
            if len(text) > 15:
                continue

            if text not in candidates:

                candidates[text] = {
                    "support": 0,
                    "confidence_values": [],
                    "structure": structure
                }

            candidates[text]["support"] += 1

            candidates[text]["confidence_values"].append(
                confidence
            )

    if not candidates:
        return None, []

    # ============================================================
    # SCORE EVERY CANDIDATE
    # ============================================================

    scored = []

    for text, info in candidates.items():

        support = info["support"]

        confidence_values = info["confidence_values"]

        confidence = (
            float(np.mean(confidence_values))
            if confidence_values
            else 0.0
        )

        structure = info["structure"]

        valid = is_valid_plate_structure(text)

        # --------------------------------------------------------
        # Consensus score
        # --------------------------------------------------------

        score = 0

        # Repeated OCR agreement is the strongest signal
        score += support * 20

        # Valid plate structure
        if valid:
            score += 50
        elif structure >= 10:
            score += 20
        else:
            score -= 20

        # OCR confidence contributes, but is NOT dominant
        score += min(confidence, 50) * 0.5

        # Length sanity
        if 8 <= len(text) <= 12:
            score += 10

        # Penalize suspiciously short strings
        if len(text) < 6:
            score -= 30

        scored.append({
            "text": text,
            "support": support,
            "confidence": confidence,
            "structure": structure,
            "valid": valid,
            "score": score
        })

    # Highest score first
    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = scored[0]

    return best, scored[:5]


# ================================================================
# IOU
# ================================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# ================================================================
# CONTAINMENT
# ================================================================

def containment_ratio(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    if area1 <= 0 or area2 <= 0:
        return 0

    return intersection / min(area1, area2)


# ================================================================
# DUPLICATE FILTER
# ================================================================

def remove_duplicate_boxes(detections):

    if len(detections) <= 1:
        return detections

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    kept = []

    for current in detections:

        duplicate = False

        for existing in kept:

            box1 = current["box"]
            box2 = existing["box"]

            iou = calculate_iou(
                box1,
                box2
            )

            containment = containment_ratio(
                box1,
                box2
            )

            cx1 = (box1[0] + box1[2]) / 2
            cy1 = (box1[1] + box1[3]) / 2

            cx2 = (box2[0] + box2[2]) / 2
            cy2 = (box2[1] + box2[3]) / 2

            center_distance = np.sqrt(
                (cx1 - cx2) ** 2 +
                (cy1 - cy2) ** 2
            )

            width = max(
                box1[2] - box1[0],
                box2[2] - box2[0]
            )

            height = max(
                box1[3] - box1[1],
                box2[3] - box2[1]
            )

            reference_size = max(
                width,
                height,
                1
            )

            center_ratio = (
                center_distance /
                reference_size
            )

            if (
                iou >= IOU_THRESHOLD
                or containment >= CONTAINMENT_THRESHOLD
                or center_ratio <= CENTER_DISTANCE_RATIO
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(current)

    return kept


# ================================================================
# DATABASE
# ================================================================

def load_database():

    database = {}

    if not os.path.exists(DATABASE_PATH):
        print("WARNING: Database file not found.")
        return database

    with open(
        DATABASE_PATH,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            plate = clean_text(
                row.get("plate_number", "")
            )

            if not plate:
                continue

            database[plate] = row

    return database


# ================================================================
# DATABASE CHECK
# ================================================================

def check_authorization(plate, database):

    plate = clean_text(plate)

    if not plate:
        return None

    record = database.get(plate)

    if record is None:
        return None

    authorized = (
        record.get("authorized", "")
        .strip()
        .upper()
    )

    status = (
        record.get("status", "")
        .strip()
        .upper()
    )

    if authorized == "YES" and status == "ACTIVE":
        return record

    return None


# ================================================================
# MAIN
# ================================================================

def main():

    if len(sys.argv) < 2:

        print(
            'Usage:\n'
            'python src\\full_anpr.py "path_to_vehicle_image.jpg"'
        )

        return

    image_path = sys.argv[1]

    print()
    print("=" * 65)
    print("FULL ANPR END-TO-END TEST")
    print("=" * 65)

    print()
    print("Input image:")
    print(image_path)

    if not os.path.exists(image_path):

        print()
        print("ERROR: Input image not found.")
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLATE_DIR,
        exist_ok=True
    )

    # ============================================================
    # LOAD IMAGE
    # ============================================================

    image = cv2.imread(image_path)

    if image is None:

        print()
        print("ERROR: Could not load image.")
        return

    height, width = image.shape[:2]

    print()
    print("Image loaded successfully.")
    print(
        f"Image size: {width} x {height}"
    )

    # ============================================================
    # LOAD YOLO
    # ============================================================

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded.")

    # ============================================================
    # DETECTION
    # ============================================================

    print()
    print("Detecting license plates...")
    print()

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": confidence
            })

    print(
        f"Raw detections: {len(detections)}"
    )

    detections = remove_duplicate_boxes(
        detections
    )

    print(
        f"Detections after duplicate removal: "
        f"{len(detections)}"
    )

    # ============================================================
    # DATABASE
    # ============================================================

    database = load_database()

    # ============================================================
    # ANNOTATED IMAGE
    # ============================================================

    annotated = image.copy()

    final_rows = []

    # ============================================================
    # PROCESS EACH PLATE
    # ============================================================

    for index, detection in enumerate(
        detections,
        start=1
    ):

        print()
        print("-" * 65)
        print(f"PLATE {index}")
        print("-" * 65)

        box = detection["box"]

        detection_confidence = detection[
            "confidence"
        ]

        x1, y1, x2, y2 = box

        crop = image[
            y1:y2,
            x1:x2
        ]

        crop_path = os.path.join(
            PLATE_DIR,
            f"plate_{index}.jpg"
        )

        cv2.imwrite(
            crop_path,
            crop
        )

        print(
            f"Detection confidence: "
            f"{detection_confidence:.2f}"
        )

        print(
            f"Crop: {crop_path}"
        )

        # ========================================================
        # OCR
        # ========================================================

        print()
        print("Running improved OCR...")

        best, candidates = run_ocr(crop)

        if best is None:

            print("OCR plate: NONE")

            final_plate = "OCR REVIEW"

            database_result = "OCR_REVIEW"

            reason = "No OCR text detected."

            ocr_confidence = 0
            support = 0
            structure = 0
            consensus_score = 0

        else:

            final_plate = best["text"]

            ocr_confidence = best["confidence"]

            support = best["support"]

            structure = best["structure"]

            consensus_score = best["score"]

            valid = best["valid"]

            print(
                f"OCR plate: {final_plate}"
            )

            print(
                f"OCR confidence: "
                f"{ocr_confidence:.2f}"
            )

            print(
                f"OCR support: {support}"
            )

            print(
                f"Structure score: {structure}"
            )

            print(
                f"Consensus score: "
                f"{consensus_score:.2f}"
            )

            print()
            print("Top OCR candidates:")

            for candidate in candidates:

                print(
                    f"  {candidate['text']} | "
                    f"support: {candidate['support']} | "
                    f"structure: {candidate['structure']} | "
                    f"confidence: "
                    f"{candidate['confidence']:.2f}"
                )

            # ====================================================
            # OCR QUALITY DECISION
            # ====================================================

            if not valid:

                final_plate = "OCR REVIEW"

                database_result = "OCR_REVIEW"

                reason = (
                    "OCR result does not match "
                    "a valid Indian plate structure."
                )

            elif support < 2:

                final_plate = "OCR REVIEW"

                database_result = "OCR_REVIEW"

                reason = (
                    "OCR agreement is too low."
                )

            elif support >= 3:

                record = check_authorization(
                    best["text"],
                    database
                )

                if record:

                    database_result = "AUTHORIZED"

                    reason = ""

                else:

                    database_result = "NOT_AUTHORIZED"

                    reason = (
                        "Plate not found as an "
                        "active authorized vehicle."
                    )

            else:

                final_plate = "OCR REVIEW"

                database_result = "OCR_REVIEW"

                reason = (
                    "Insufficient OCR consensus."
                )

        # ========================================================
        # PRINT DATABASE RESULT
        # ========================================================

        print()

        if database_result == "AUTHORIZED":

            record = check_authorization(
                final_plate,
                database
            )

            print(
                "Database result: AUTHORIZED"
            )

            if record:

                print(
                    f"Owner: "
                    f"{record.get('owner_name', '')}"
                )

                print(
                    f"Vehicle type: "
                    f"{record.get('vehicle_type', '')}"
                )

        elif database_result == "NOT_AUTHORIZED":

            print(
                "Database result: NOT_AUTHORIZED"
            )

        else:

            print(
                "Database result: OCR_REVIEW"
            )

            print(
                f"Reason: {reason}"
            )

        # ========================================================
        # ANNOTATION
        # ========================================================

        if database_result == "AUTHORIZED":

            label = (
                f"{final_plate} | AUTHORIZED"
            )

        elif database_result == "NOT_AUTHORIZED":

            label = (
                f"{final_plate} | NOT AUTHORIZED"
            )

        else:

            label = (
                "OCR REVIEW"
            )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ========================================================
        # SAVE RESULT ROW
        # ========================================================

        final_rows.append({
            "plate": final_plate,
            "detection_confidence":
                round(detection_confidence, 4),
            "ocr_confidence":
                round(ocr_confidence, 2),
            "support": support,
            "structure_score": structure,
            "consensus_score":
                round(consensus_score, 2),
            "authorization":
                database_result,
            "reason": reason
        })

    # ============================================================
    # FINAL RESULT
    # ============================================================

    print()
    print("=" * 65)
    print("FINAL ANPR RESULT")
    print("=" * 65)

    for row in final_rows:

        print()

        print(
            f"Plate: {row['plate']}"
        )

        print(
            f"Detection confidence: "
            f"{row['detection_confidence']:.2f}"
        )

        print(
            f"OCR confidence: "
            f"{row['ocr_confidence']:.2f}"
        )

        print(
            f"Support: {row['support']}"
        )

        print(
            f"Structure score: "
            f"{row['structure_score']}"
        )

        print(
            f"Authorization: "
            f"{row['authorization']}"
        )

    # ============================================================
    # SAVE ANNOTATED IMAGE
    # ============================================================

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    annotated_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_result.jpg"
    )

    cv2.imwrite(
        annotated_path,
        annotated
    )

    # ============================================================
    # SAVE CSV
    # ============================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_result.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "plate",
                "detection_confidence",
                "ocr_confidence",
                "support",
                "structure_score",
                "consensus_score",
                "authorization",
                "reason"
            ]
        )

        writer.writeheader()

        writer.writerows(
            final_rows
        )

    print()
    print("Annotated image saved:")
    print(annotated_path)

    print()
    print("CSV result saved:")
    print(csv_path)

    print()
    print("=" * 65)
    print("END-TO-END TEST COMPLETED")
    print("=" * 65)

    print()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()