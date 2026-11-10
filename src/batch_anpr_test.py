import os
import csv
import subprocess
import sys
import re


# ================================================================
# SETTINGS
# ================================================================

TEST_DIR = r"D:\ANPR_Project\dataset\anpr_yolo\images\test"

OUTPUT_DIR = r"D:\ANPR_Project\outputs\batch_test"

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "batch_anpr_results.csv"
)

FULL_ANPR_SCRIPT = r"D:\ANPR_Project\src\full_anpr.py"


# ================================================================
# MAIN
# ================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    images = []

    for filename in os.listdir(TEST_DIR):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            images.append(filename)

    images.sort()

    print()
    print("=" * 65)
    print("BATCH ANPR TEST")
    print("=" * 65)

    print()
    print(f"Test images found: {len(images)}")

    if not images:

        print("ERROR: No test images found.")
        return

    results = []

    for index, filename in enumerate(
        images,
        start=1
    ):

        image_path = os.path.join(
            TEST_DIR,
            filename
        )

        print()
        print(
            f"[{index}/{len(images)}] {filename}"
        )

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    FULL_ANPR_SCRIPT,
                    image_path
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )

            output = process.stdout

            # ----------------------------------------------------
            # Count detected plates
            # ----------------------------------------------------

            match = re.search(
                r"Detections after duplicate removal:\s*(\d+)",
                output
            )

            detection_count = (
                int(match.group(1))
                if match
                else 0
            )

            # ----------------------------------------------------
            # Count final authorization results
            # ----------------------------------------------------

            authorized = output.count(
                "Authorization: AUTHORIZED"
            )

            not_authorized = output.count(
                "Authorization: NOT_AUTHORIZED"
            )

            ocr_review = output.count(
                "Authorization: OCR_REVIEW"
            )

            # ----------------------------------------------------
            # OCR plates
            # ----------------------------------------------------

            plates = re.findall(
                r"Plate:\s*(.+)",
                output
            )

            plates = [
                p.strip()
                for p in plates
            ]

            # ----------------------------------------------------
            # Save result
            # ----------------------------------------------------

            results.append({
                "filename": filename,
                "detections": detection_count,
                "authorized": authorized,
                "not_authorized": not_authorized,
                "ocr_review": ocr_review,
                "plates": " | ".join(plates)
            })

            print(
                f"  Detections: {detection_count}"
            )

            print(
                f"  Authorized: {authorized}"
            )

            print(
                f"  Not authorized: {not_authorized}"
            )

            print(
                f"  OCR review: {ocr_review}"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            results.append({
                "filename": filename,
                "detections": 0,
                "authorized": 0,
                "not_authorized": 0,
                "ocr_review": 0,
                "plates": "ERROR"
            })

    # ============================================================
    # SAVE CSV
    # ============================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "detections",
                "authorized",
                "not_authorized",
                "ocr_review",
                "plates"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    # ============================================================
    # SUMMARY
    # ============================================================

    total_images = len(results)

    total_detections = sum(
        r["detections"]
        for r in results
    )

    total_authorized = sum(
        r["authorized"]
        for r in results
    )

    total_not_authorized = sum(
        r["not_authorized"]
        for r in results
    )

    total_review = sum(
        r["ocr_review"]
        for r in results
    )

    print()
    print("=" * 65)
    print("BATCH ANPR SUMMARY")
    print("=" * 65)

    print()
    print(
        f"Total test images: {total_images}"
    )

    print(
        f"Total detected plates: {total_detections}"
    )

    print(
        f"AUTHORIZED: {total_authorized}"
    )

    print(
        f"NOT_AUTHORIZED: {total_not_authorized}"
    )

    print(
        f"OCR_REVIEW: {total_review}"
    )

    print()
    print("Results saved:")
    print(CSV_PATH)

    print()
    print("=" * 65)
    print("BATCH TEST COMPLETED")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()