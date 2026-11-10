import os
import shutil
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"D:\ANPR_Project"

CONSENSUS_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "ocr_results",
    "consensus_ocr_results.csv"
)

PLATES_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "plates"
)

REVIEW_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "ocr_review"
)

REVIEW_CSV = os.path.join(
    REVIEW_DIR,
    "ocr_review.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOTAL_REVIEW = 30
ACCEPT_COUNT = 14


# ============================================================
# CREATE REVIEW DIRECTORY
# ============================================================

os.makedirs(REVIEW_DIR, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(CONSENSUS_FILE):
    print("ERROR: Consensus file not found:")
    print(CONSENSUS_FILE)
    raise SystemExit(1)

if not os.path.exists(PLATES_DIR):
    print("ERROR: Plate directory not found:")
    print(PLATES_DIR)
    raise SystemExit(1)


# ============================================================
# READ CONSENSUS RESULTS
# ============================================================

print("Reading consensus OCR results...")
print(CONSENSUS_FILE)

df = pd.read_csv(CONSENSUS_FILE)

print(f"Total records: {len(df)}")


# ============================================================
# CLEAN DATA
# ============================================================

df["decision"] = df["decision"].fillna("").astype(str)
df["filename"] = df["filename"].fillna("").astype(str)
df["recommended_plate"] = (
    df["recommended_plate"]
    .fillna("")
    .astype(str)
)

for column in [
    "confidence",
    "support_count",
    "structure_score",
    "consensus_score"
]:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)


# ============================================================
# GET ACCEPTED RESULTS
# ============================================================

accepted = df[
    df["decision"].str.upper() == "ACCEPT"
].copy()

accepted = accepted.sort_values(
    by="consensus_score",
    ascending=False
)


# ============================================================
# GET REVIEW RESULTS
# ============================================================

review = df[
    df["decision"].str.upper() == "REVIEW"
].copy()

review = review.sort_values(
    by="consensus_score",
    ascending=False
)


# ============================================================
# SELECT 30 PLATES
# ============================================================

selected = []

# First include accepted results
accepted_to_use = accepted.head(ACCEPT_COUNT)

selected.append(accepted_to_use)

# Calculate remaining
remaining = TOTAL_REVIEW - len(accepted_to_use)

# Add highest-scoring REVIEW results
if remaining > 0:
    selected.append(
        review.head(remaining)
    )


# Combine
review_df = pd.concat(
    selected,
    ignore_index=True
)

review_df = review_df.head(TOTAL_REVIEW).copy()


# ============================================================
# ADD MANUAL VALIDATION COLUMNS
# ============================================================

review_df["verified_plate"] = ""
review_df["is_correct"] = ""
review_df["notes"] = ""


# ============================================================
# COPY IMAGES
# ============================================================

print()
print("Copying plate images...")

copied = 0
missing = 0

for filename in review_df["filename"]:

    source = os.path.join(
        PLATES_DIR,
        filename
    )

    destination = os.path.join(
        REVIEW_DIR,
        filename
    )

    if os.path.exists(source):

        shutil.copy2(
            source,
            destination
        )

        copied += 1

    else:

        print("Missing image:")
        print(source)

        missing += 1


# ============================================================
# IMAGE STATUS
# ============================================================

review_df["image_available"] = review_df["filename"].apply(
    lambda filename:
        os.path.exists(
            os.path.join(
                REVIEW_DIR,
                filename
            )
        )
)


# ============================================================
# COLUMN ORDER
# ============================================================

columns = [
    "filename",
    "recommended_plate",
    "confidence",
    "support_count",
    "structure_score",
    "consensus_score",
    "decision",
    "verified_plate",
    "is_correct",
    "notes",
    "image_available"
]

review_df = review_df[columns]


# ============================================================
# SAVE CSV
# ============================================================

review_df.to_csv(
    REVIEW_CSV,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("OCR REVIEW SET CREATED")
print("=" * 60)

print(f"Total plates selected : {len(review_df)}")
print(f"Images copied         : {copied}")
print(f"Images missing        : {missing}")

print()
print("Review folder:")
print(REVIEW_DIR)

print()
print("Review CSV:")
print(REVIEW_CSV)

print()
print("Open the CSV in Excel and manually verify the images.")
print("=" * 60)