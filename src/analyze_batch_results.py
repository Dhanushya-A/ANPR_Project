import pandas as pd
import os

CSV_PATH = r"D:\ANPR_Project\outputs\batch_test\batch_anpr_results.csv"

print("=" * 70)
print("BATCH ANPR RESULT ANALYSIS")
print("=" * 70)

if not os.path.exists(CSV_PATH):
    print("\nERROR: Batch results CSV not found.")
    print(CSV_PATH)
    raise SystemExit

df = pd.read_csv(CSV_PATH)

print(f"\nTotal records: {len(df)}")

# ------------------------------------------------------------
# COLUMN CHECK
# ------------------------------------------------------------

required_columns = [
    "filename",
    "detections",
    "authorized",
    "not_authorized",
    "ocr_review",
    "plates"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    print("\nERROR: Missing columns:")
    print(missing)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise SystemExit

# ------------------------------------------------------------
# TOTALS
# ------------------------------------------------------------

total_images = len(df)
total_detections = int(df["detections"].sum())
authorized = int(df["authorized"].sum())
not_authorized = int(df["not_authorized"].sum())
ocr_review = int(df["ocr_review"].sum())
no_detection = int((df["detections"] == 0).sum())

print("\n" + "-" * 70)
print("OVERALL SUMMARY")
print("-" * 70)

print(f"Total test images       : {total_images}")
print(f"Images with detections  : {total_images - no_detection}")
print(f"Images without detection: {no_detection}")
print(f"Total detected plates   : {total_detections}")
print(f"AUTHORIZED              : {authorized}")
print(f"NOT_AUTHORIZED          : {not_authorized}")
print(f"OCR_REVIEW              : {ocr_review}")

# ------------------------------------------------------------
# DETECTION RATE
# ------------------------------------------------------------

detection_rate = (
    (total_images - no_detection) / total_images * 100
    if total_images > 0 else 0
)

print(f"\nDetection coverage      : {detection_rate:.2f}%")

# ------------------------------------------------------------
# AUTHORIZED RECORDS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("AUTHORIZED IMAGES")
print("-" * 70)

authorized_rows = df[df["authorized"] > 0]

if len(authorized_rows) == 0:
    print("No authorized records found.")
else:
    print(
        authorized_rows[
            ["filename", "detections", "authorized", "plates"]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# NOT AUTHORIZED
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("NOT AUTHORIZED SUMMARY")
print("-" * 70)

not_auth_rows = df[df["not_authorized"] > 0]

print(f"Images marked NOT_AUTHORIZED: {len(not_auth_rows)}")

if len(not_auth_rows) > 0:
    print("\nFirst 20:")
    print(
        not_auth_rows[
            ["filename", "detections", "not_authorized", "plates"]
        ].head(20).to_string(index=False)
    )

# ------------------------------------------------------------
# OCR REVIEW
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("OCR REVIEW SUMMARY")
print("-" * 70)

review_rows = df[df["ocr_review"] > 0]

print(f"Images requiring OCR review: {len(review_rows)}")

if len(review_rows) > 0:
    print("\nFirst 30 OCR review records:")
    print(
        review_rows[
            ["filename", "detections", "ocr_review", "plates"]
        ].head(30).to_string(index=False)
    )

# ------------------------------------------------------------
# NO DETECTION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("NO DETECTION IMAGES")
print("-" * 70)

no_detection_rows = df[df["detections"] == 0]

if len(no_detection_rows) == 0:
    print("All images had at least one detection.")
else:
    for filename in no_detection_rows["filename"]:
        print(filename)

# ------------------------------------------------------------
# MULTIPLE DETECTIONS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("MULTIPLE DETECTION IMAGES")
print("-" * 70)

multiple_rows = df[df["detections"] > 1]

print(f"Images with multiple detected plates: {len(multiple_rows)}")

if len(multiple_rows) > 0:
    print(
        multiple_rows[
            ["filename", "detections", "authorized",
             "not_authorized", "ocr_review", "plates"]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# SAVE CLEAN SUMMARY
# ------------------------------------------------------------

SUMMARY_CSV = (
    r"D:\ANPR_Project\outputs\batch_test"
    r"\batch_analysis_summary.csv"
)

summary_data = {
    "total_images": [total_images],
    "images_with_detections": [total_images - no_detection],
    "images_without_detection": [no_detection],
    "total_detected_plates": [total_detections],
    "authorized": [authorized],
    "not_authorized": [not_authorized],
    "ocr_review": [ocr_review],
    "detection_coverage_percent": [round(detection_rate, 2)]
}

pd.DataFrame(summary_data).to_csv(
    SUMMARY_CSV,
    index=False
)

print("\n" + "-" * 70)
print("SUMMARY FILE")
print("-" * 70)

print(SUMMARY_CSV)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)