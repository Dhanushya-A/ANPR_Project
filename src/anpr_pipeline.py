import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"D:\ANPR_Project"

OCR_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "ocr_results",
    "consensus_ocr_results.csv"
)

DATABASE_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "vehicles.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "anpr_results.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZE PLATE
# ============================================================

def normalize_plate(plate):

    return (
        str(plate)
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .strip()
    )


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    if not os.path.exists(DATABASE_FILE):

        print("ERROR: Vehicle database not found:")
        print(DATABASE_FILE)

        raise SystemExit(1)

    database = pd.read_csv(
        DATABASE_FILE
    )

    database["normalized_plate"] = (
        database["plate_number"]
        .apply(normalize_plate)
    )

    return database


# ============================================================
# CHECK AUTHORIZATION
# ============================================================

def check_authorization(
    plate_number,
    database
):

    normalized = normalize_plate(
        plate_number
    )

    if normalized == "":
        return "OCR_FAILED"

    matches = database[
        (database["normalized_plate"] == normalized) &
        (
            database["authorized"]
            .astype(str)
            .str.upper()
            == "YES"
        ) &
        (
            database["status"]
            .astype(str)
            .str.upper()
            == "ACTIVE"
        )
    ]

    if len(matches) > 0:

        return "AUTHORIZED"

    return "NOT_AUTHORIZED"


# ============================================================
# MAIN PIPELINE
# ============================================================

print("=" * 70)
print("AUTOMATIC NUMBER PLATE RECOGNITION PIPELINE")
print("=" * 70)


# ------------------------------------------------------------
# Check OCR file
# ------------------------------------------------------------

if not os.path.exists(OCR_FILE):

    print()
    print("ERROR: OCR consensus file not found:")
    print(OCR_FILE)

    raise SystemExit(1)


# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

print()
print("Loading OCR results...")

ocr_df = pd.read_csv(
    OCR_FILE
)

database = load_database()

print(f"OCR records       : {len(ocr_df)}")
print(f"Database records  : {len(database)}")


# ============================================================
# PROCESS OCR RESULTS
# ============================================================

results = []

for _, row in ocr_df.iterrows():

    filename = row["filename"]

    plate = row["recommended_plate"]

    confidence = row["confidence"]

    support = row["support_count"]

    structure = row["structure_score"]

    consensus = row["consensus_score"]

    decision = row["decision"]

    authorization = check_authorization(
        plate,
        database
    )

    results.append({

        "filename": filename,

        "recognized_plate": plate,

        "ocr_confidence": confidence,

        "support_count": support,

        "structure_score": structure,

        "consensus_score": consensus,

        "ocr_decision": decision,

        "authorization": authorization
    })


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ANPR PIPELINE RESULTS")
print("=" * 70)

print(
    f"Total plates processed : "
    f"{len(results_df)}"
)

print()

print(
    "Authorization results:"
)

print(
    results_df["authorization"]
    .value_counts()
    .to_string()
)


# ============================================================
# SHOW AUTHORIZED PLATES
# ============================================================

authorized = results_df[
    results_df["authorization"]
    == "AUTHORIZED"
]

print()
print("=" * 70)
print("AUTHORIZED PLATES")
print("=" * 70)

if len(authorized) == 0:

    print("No authorized plates detected.")

else:

    print(
        authorized[
            [
                "filename",
                "recognized_plate",
                "ocr_confidence",
                "support_count"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SHOW NOT AUTHORIZED
# ============================================================

not_authorized = results_df[
    results_df["authorization"]
    == "NOT_AUTHORIZED"
]

print()
print("=" * 70)
print("NOT AUTHORIZED")
print("=" * 70)

print(
    f"Count: {len(not_authorized)}"
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("Results saved to:")
print(OUTPUT_FILE)
print("=" * 70)