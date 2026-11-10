import cv2
import os
import numpy as np


# ============================================================
# ANPR - NUMBER PLATE PREPROCESSING
# ============================================================

# Input folder containing detected plate crops
INPUT_DIR = r"D:\ANPR_Project\outputs\plates"

# Output folder for OCR-ready images
OUTPUT_DIR = r"D:\ANPR_Project\outputs\preprocessed"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GET INPUT IMAGES
# ============================================================

image_files = [
    f
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

print("=" * 60)
print("ANPR NUMBER PLATE PREPROCESSING")
print("=" * 60)

print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Plate crops  : {len(image_files)}")
print("=" * 60)


# ============================================================
# PROCESS EACH PLATE
# ============================================================

processed_count = 0
failed_count = 0


for image_name in image_files:

    input_path = os.path.join(
        INPUT_DIR,
        image_name
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(input_path)

    if image is None:
        print(f"[ERROR] Could not read: {image_name}")
        failed_count += 1
        continue

    # --------------------------------------------------------
    # 1. CONVERT TO GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 2. RESIZE
    # --------------------------------------------------------
    # Increase character size for OCR.

    scale = 3

    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # 3. DENOISING
    # --------------------------------------------------------
    # Reduce small image noise.

    denoised = cv2.GaussianBlur(
        resized,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # 4. CONTRAST ENHANCEMENT
    # --------------------------------------------------------
    # CLAHE improves local contrast.

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    # --------------------------------------------------------
    # 5. OTSU THRESHOLDING
    # --------------------------------------------------------
    # Converts the plate into a binary image.

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 6. ADAPTIVE THRESHOLDING
    # --------------------------------------------------------
    # Useful when lighting is not uniform.

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # --------------------------------------------------------
    # 7. SHARPENING
    # --------------------------------------------------------
    # Improve character edges.

    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpening_kernel
    )

    # --------------------------------------------------------
    # 8. MORPHOLOGICAL CLEANING
    # --------------------------------------------------------
    # Helps remove small noise and connect character regions.

    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    morphological = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        morphology_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # CREATE BASE FILE NAME
    # --------------------------------------------------------

    base_name = os.path.splitext(
        image_name
    )[0]

    # --------------------------------------------------------
    # SAVE GRAYSCALE IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_gray.jpg"
        ),
        resized
    )

    # --------------------------------------------------------
    # SAVE ENHANCED IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_enhanced.jpg"
        ),
        enhanced
    )

    # --------------------------------------------------------
    # SAVE OTSU IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_otsu.jpg"
        ),
        otsu
    )

    # --------------------------------------------------------
    # SAVE ADAPTIVE IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_adaptive.jpg"
        ),
        adaptive
    )

    # --------------------------------------------------------
    # SAVE SHARPENED IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_sharp.jpg"
        ),
        sharpened
    )

    # --------------------------------------------------------
    # SAVE MORPHOLOGICAL IMAGE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_morph.jpg"
        ),
        morphological
    )

    processed_count += 1

    print(
        f"[OK] {image_name}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Successfully processed : {processed_count}")
print(f"Failed                 : {failed_count}")
print(f"Total input images     : {len(image_files)}")

print()
print("Preprocessed images saved to:")
print(OUTPUT_DIR)

print("=" * 60)