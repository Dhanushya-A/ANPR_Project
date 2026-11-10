import os
import re
import csv
import cv2
import pytesseract

# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"D:\ANPR_Project\outputs\plates"
OUTPUT_DIR = r"D:\ANPR_Project\outputs\ocr_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "ocr_results.csv"
)

TXT_FILE = os.path.join(
    OUTPUT_DIR,
    "ocr_results.txt"
)

# ============================================================
# COMMON INDIAN STATE / UT CODES
# ============================================================

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD",
    "DL", "DN", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB"
}

# ============================================================
# OCR SETTINGS
# ============================================================

PSM_MODES = [6, 7, 8, 13]

WHITELIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    text = text.upper()

    # Remove spaces, hyphens, punctuation, etc.
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# CREATE IMAGE VARIANTS
# ============================================================

def create_variants(image):

    variants = {}

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    variants["original"] = image

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variants["gray"] = gray

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    enlarged = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    variants["enlarged"] = enlarged

    # --------------------------------------------------------
    # CLAHE contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        enlarged
    )

    variants["enhanced"] = enhanced

    # --------------------------------------------------------
    # Gaussian blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    variants["blurred"] = blurred

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants["otsu"] = otsu

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants["adaptive"] = adaptive

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        cv2.GaussianBlur(
            enhanced,
            (0, 0),
            3
        ),
        -0.5,
        0
    )

    variants["sharpened"] = sharpened

    # --------------------------------------------------------
    # Morphological closing
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    morph = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        kernel
    )

    variants["morph"] = morph

    return variants


# ============================================================
# INDIAN PLATE STRUCTURE SCORE
# ============================================================

def plate_structure_score(text):

    text = clean_text(text)

    if not text:
        return 0

    score = 0

    length = len(text)

    # --------------------------------------------------------
    # Reasonable Indian plate length
    # --------------------------------------------------------

    if 8 <= length <= 12:
        score += 3

    elif 7 <= length <= 13:
        score += 1

    # --------------------------------------------------------
    # Starts with two letters
    # --------------------------------------------------------

    if (
        len(text) >= 2
        and text[0].isalpha()
        and text[1].isalpha()
    ):
        score += 3

        # Recognized state code
        if text[:2] in STATE_CODES:
            score += 4

    # --------------------------------------------------------
    # Contains digits
    # --------------------------------------------------------

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    if digit_count >= 2:
        score += 2

    # --------------------------------------------------------
    # Contains letters
    # --------------------------------------------------------

    letter_count = sum(
        char.isalpha()
        for char in text
    )

    if letter_count >= 2:
        score += 1

    # --------------------------------------------------------
    # Avoid obviously bad OCR
    # --------------------------------------------------------

    if length > 15:
        score -= 5

    return score


# ============================================================
# OCR WITH CONFIDENCE
# ============================================================

def run_ocr(image, psm):

    config = (
        f"--oem 3 --psm {psm} "
        f"-c tessedit_char_whitelist={WHITELIST}"
    )

    try:

        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT
        )

    except Exception as e:

        print("OCR error:", e)

        return "", 0.0

    texts = []
    confidences = []

    for i in range(len(data["text"])):

        raw_text = data["text"][i]

        text = clean_text(raw_text)

        if text:

            texts.append(text)

            try:
                confidence = float(
                    data["conf"][i]
                )

                if confidence >= 0:
                    confidences.append(
                        confidence
                    )

            except ValueError:
                pass

    combined_text = "".join(texts)

    if confidences:

        avg_confidence = (
            sum(confidences)
            / len(confidences)
        )

    else:

        avg_confidence = 0.0

    return combined_text, avg_confidence


# ============================================================
# PROCESS ONE PLATE
# ============================================================

def recognize_plate(filepath):

    image = cv2.imread(filepath)

    if image is None:

        return {
            "best_text": "",
            "best_confidence": 0,
            "best_score": 0,
            "candidates": []
        }

    variants = create_variants(image)

    candidates = []

    # --------------------------------------------------------
    # Run OCR on every variant and PSM
    # --------------------------------------------------------

    for variant_name, variant_image in variants.items():

        for psm in PSM_MODES:

            text, confidence = run_ocr(
                variant_image,
                psm
            )

            if not text:
                continue

            structure_score = (
                plate_structure_score(text)
            )

            # Combined score
            total_score = (
                structure_score * 10
                + confidence
            )

            candidates.append(
                {
                    "text": text,
                    "variant": variant_name,
                    "psm": psm,
                    "confidence": confidence,
                    "structure_score": structure_score,
                    "total_score": total_score
                }
            )

    # --------------------------------------------------------
    # Sort best candidate first
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["total_score"],
        reverse=True
    )

    if candidates:

        best = candidates[0]

        return {
            "best_text": best["text"],
            "best_confidence": best["confidence"],
            "best_score": best["total_score"],
            "candidates": candidates
        }

    return {
        "best_text": "",
        "best_confidence": 0,
        "best_score": 0,
        "candidates": []
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANPR - FINAL OCR ENGINE TEST")
    print("=" * 70)

    if not os.path.exists(INPUT_DIR):

        print()
        print("ERROR: Plate directory not found:")
        print(INPUT_DIR)

        return

    files = sorted(
        [
            f
            for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    print()
    print("Original plate crops found:", len(files))

    print()
    print("The OCR engine will test:")
    print("  - 8 image variants")
    print("  - 4 Tesseract PSM modes")
    print("  - Multiple OCR candidates")
    print()

    # --------------------------------------------------------
    # Process all plate crops
    # --------------------------------------------------------

    csv_rows = []

    with open(
        TXT_FILE,
        "w",
        encoding="utf-8"
    ) as txt:

        txt.write(
            "ANPR OCR RESULTS\n"
        )

        txt.write(
            "=" * 70 + "\n\n"
        )

        for index, filename in enumerate(
            files,
            start=1
        ):

            filepath = os.path.join(
                INPUT_DIR,
                filename
            )

            result = recognize_plate(
                filepath
            )

            best_text = result["best_text"]

            confidence = result[
                "best_confidence"
            ]

            score = result[
                "best_score"
            ]

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"{index:03d}/{len(files):03d} "
                f"{filename}"
            )

            print(
                f"      Best OCR : {best_text}"
            )

            print(
                f"      Confidence: {confidence:.2f}"
            )

            print(
                f"      Score     : {score:.2f}"
            )

            # ------------------------------------------------
            # Text file
            # ------------------------------------------------

            txt.write(
                f"{filename}\n"
            )

            txt.write(
                f"Best OCR: {best_text}\n"
            )

            txt.write(
                f"Confidence: {confidence:.2f}\n"
            )

            txt.write(
                f"Score: {score:.2f}\n"
            )

            txt.write(
                "Candidates:\n"
            )

            for candidate in result[
                "candidates"
            ][:10]:

                txt.write(
                    f"  "
                    f"{candidate['text']} | "
                    f"{candidate['variant']} | "
                    f"PSM {candidate['psm']} | "
                    f"conf={candidate['confidence']:.2f} | "
                    f"structure={candidate['structure_score']}\n"
                )

            txt.write("\n")

            # ------------------------------------------------
            # CSV row
            # ------------------------------------------------

            csv_rows.append(
                {
                    "image": filename,
                    "plate_text": best_text,
                    "confidence": round(
                        confidence,
                        2
                    ),
                    "score": round(
                        score,
                        2
                    )
                }
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        fieldnames = [
            "image",
            "plate_text",
            "confidence",
            "score"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    recognized = sum(
        1
        for row in csv_rows
        if row["plate_text"]
    )

    print()
    print("=" * 70)
    print("OCR PROCESSING COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Plate crops processed : {len(files)}"
    )

    print(
        f"Plates with OCR text  : {recognized}"
    )

    print(
        f"Plates with no text   : "
        f"{len(files) - recognized}"
    )

    print()
    print("CSV results:")
    print(CSV_FILE)

    print()
    print("Detailed results:")
    print(TXT_FILE)


if __name__ == "__main__":
    main()