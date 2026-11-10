import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"D:\ANPR_Project"

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "ocr_review",
    "ocr_review.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "outputs",
    "ocr_review",
    "ocr_evaluation.csv"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    print("ERROR: Review CSV not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("OCR EVALUATION")
print("=" * 60)

print(f"Total review records: {len(df)}")


# ============================================================
# CLEAN TEXT
# ============================================================

df["recommended_plate"] = (
    df["recommended_plate"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.replace(" ", "", regex=False)
)

df["verified_plate"] = (
    df["verified_plate"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.replace(" ", "", regex=False)
)

df["is_correct"] = (
    df["is_correct"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# CHECK FOR MISSING VERIFICATION
# ============================================================

missing = df[
    df["verified_plate"] == ""
].copy()

if len(missing) > 0:

    print()
    print("WARNING")
    print("-" * 60)
    print(
        f"{len(missing)} records do not have "
        "a verified_plate value."
    )

    print()
    print("Please complete all verified_plate values")
    print("before using the accuracy results.")

    print()
    print("Missing verification:")
    print(
        missing[
            ["filename", "recommended_plate"]
        ].to_string(index=False)
    )

    raise SystemExit(1)


# ============================================================
# EXACT MATCH
# ============================================================

df["exact_match"] = (
    df["recommended_plate"]
    == df["verified_plate"]
)


# ============================================================
# CHARACTER ACCURACY
# ============================================================

def levenshtein_distance(a, b):

    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1]

            if char_a != char_b:
                substitution += 1

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


def character_accuracy(predicted, actual):

    if actual == "":
        return 0.0

    distance = levenshtein_distance(
        predicted,
        actual
    )

    accuracy = (
        1 -
        distance / max(
            len(predicted),
            len(actual)
        )
    )

    return max(0.0, accuracy)


df["character_accuracy"] = df.apply(
    lambda row:
        character_accuracy(
            row["recommended_plate"],
            row["verified_plate"]
        ),
    axis=1
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CALCULATE SUMMARY
# ============================================================

total = len(df)

exact_matches = int(
    df["exact_match"].sum()
)

exact_accuracy = (
    exact_matches / total * 100
)

character_accuracy_average = (
    df["character_accuracy"].mean() * 100
)


# ============================================================
# CHARACTER ERRORS
# ============================================================

total_edit_distance = 0
total_actual_characters = 0

for _, row in df.iterrows():

    predicted = row["recommended_plate"]
    actual = row["verified_plate"]

    distance = levenshtein_distance(
        predicted,
        actual
    )

    total_edit_distance += distance
    total_actual_characters += len(actual)


if total_actual_characters > 0:

    character_error_rate = (
        total_edit_distance /
        total_actual_characters *
        100
    )

else:

    character_error_rate = 0


# ============================================================
# CORRECT / INCORRECT
# ============================================================

correct_rows = df[
    df["exact_match"]
]

incorrect_rows = df[
    ~df["exact_match"]
]


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("OCR ACCURACY RESULTS")
print("=" * 60)

print(f"Total plates tested       : {total}")
print(f"Exact matches             : {exact_matches}")
print(f"Incorrect results         : {len(incorrect_rows)}")

print()
print(
    f"Exact-match accuracy      : "
    f"{exact_accuracy:.2f}%"
)

print(
    f"Average character accuracy: "
    f"{character_accuracy_average:.2f}%"
)

print(
    f"Character error rate      : "
    f"{character_error_rate:.2f}%"
)


# ============================================================
# SHOW INCORRECT RESULTS
# ============================================================

if len(incorrect_rows) > 0:

    print()
    print("=" * 60)
    print("INCORRECT OCR RESULTS")
    print("=" * 60)

    print(
        incorrect_rows[
            [
                "filename",
                "recommended_plate",
                "verified_plate",
                "character_accuracy",
                "notes"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SHOW CORRECT RESULTS
# ============================================================

print()
print("=" * 60)
print("CORRECT OCR RESULTS")
print("=" * 60)

print(
    correct_rows[
        [
            "filename",
            "recommended_plate",
            "verified_plate"
        ]
    ].to_string(index=False)
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("Evaluation file saved:")
print(OUTPUT_FILE)
print("=" * 60)