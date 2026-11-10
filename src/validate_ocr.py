import os
import re
import csv

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"D:\ANPR_Project\outputs\ocr_results\ocr_results.csv"

OUTPUT_FILE = (
    r"D:\ANPR_Project\outputs\ocr_results"
    r"\validated_ocr_results.csv"
)

# ============================================================
# INDIAN STATE / UNION TERRITORY CODES
# ============================================================

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD",
    "DL", "DN", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB"
}

# ============================================================
# COMMON OCR CHARACTER CONFUSIONS
# ============================================================

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8"
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B"
}


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# CORRECT A SPECIFIC POSITION
# ============================================================

def correct_position(
    text,
    start,
    end,
    mapping
):
    """
    Apply OCR corrections only inside
    a specified character range.
    """

    chars = list(text)

    for i in range(
        start,
        min(end, len(chars))
    ):

        if chars[i] in mapping:
            chars[i] = mapping[chars[i]]

    return "".join(chars)


# ============================================================
# TRY INDIAN PLATE FORMATS
# ============================================================

def validate_plate(text):

    text = clean_text(text)

    if not text:
        return "", "EMPTY", False

    # --------------------------------------------------------
    # Standard Indian registration:
    #
    # SS + RTO(1-2 digits) + SERIES(1-3 letters)
    # + NUMBER(1-4 digits)
    #
    # Examples:
    # KA04MS7912
    # TN72BB4165
    # MH04DW7092
    # --------------------------------------------------------

    standard_pattern = (
        r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$"
    )

    match = re.match(
        standard_pattern,
        text
    )

    if match:

        state = match.group(1)

        if state in STATE_CODES:

            return text, "VALID", True

        return text, "UNKNOWN_STATE", False

    # --------------------------------------------------------
    # Bharat Series:
    #
    # 2 digits + BH + 4 digits + 1-2 letters
    # --------------------------------------------------------

    bh_pattern = (
        r"^(\d{2})BH(\d{4})([A-Z]{1,2})$"
    )

    if re.match(
        bh_pattern,
        text
    ):

        return text, "BH_SERIES", True

    # --------------------------------------------------------
    # Try OCR corrections based on position
    # --------------------------------------------------------

    # We need at least:
    # 2 state letters + something else

    if len(text) >= 5:

        corrected = text

        # First two positions should be letters.
        corrected = correct_position(
            corrected,
            0,
            2,
            DIGIT_TO_LETTER
        )

        # Next two positions are most likely
        # RTO digits.
        corrected = correct_position(
            corrected,
            2,
            4,
            LETTER_TO_DIGIT
        )

        # Try standard format again.
        match = re.match(
            standard_pattern,
            corrected
        )

        if match:

            state = match.group(1)

            if state in STATE_CODES:

                return (
                    corrected,
                    "CORRECTED",
                    True
                )

    # --------------------------------------------------------
    # Too short / invalid structure
    # --------------------------------------------------------

    if len(text) < 7:

        return (
            text,
            "TOO_SHORT",
            False
        )

    return (
        text,
        "REVIEW",
        False
    )


# ============================================================
# CONFIDENCE CATEGORY
# ============================================================

def confidence_category(confidence):

    try:
        confidence = float(confidence)
    except:
        confidence = 0

    if confidence >= 80:
        return "HIGH"

    if confidence >= 50:
        return "MEDIUM"

    if confidence > 0:
        return "LOW"

    return "UNKNOWN"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANPR - OCR VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print()
        print("ERROR: OCR CSV not found:")
        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print()
    print(
        "OCR records found:",
        len(rows)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_count = 0
    corrected_count = 0
    review_count = 0
    short_count = 0
    empty_count = 0

    output_rows = []

    # --------------------------------------------------------
    # Validate each plate
    # --------------------------------------------------------

    for row in rows:

        filename = row.get(
            "image",
            ""
        )

        original = clean_text(
            row.get(
                "plate_text",
                ""
            )
        )

        confidence = row.get(
            "confidence",
            "0"
        )

        validated, status, is_valid = (
            validate_plate(original)
        )

        confidence_level = (
            confidence_category(
                confidence
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if status == "VALID":

            valid_count += 1

        elif status == "CORRECTED":

            corrected_count += 1

        elif status == "REVIEW":

            review_count += 1

        elif status == "TOO_SHORT":

            short_count += 1

        elif status == "EMPTY":

            empty_count += 1

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        output_rows.append(
            {
                "image": filename,
                "ocr_text": original,
                "validated_text": validated,
                "status": status,
                "confidence": confidence,
                "confidence_level": confidence_level
            }
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"{filename:35s} "
            f"{original:18s} -> "
            f"{validated:18s} "
            f"[{status}]"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "ocr_text",
            "validated_text",
            "status",
            "confidence",
            "confidence_level"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OCR VALIDATION COMPLETED")
    print("=" * 70)

    print()
    print(
        "Total records       :",
        len(rows)
    )

    print(
        "Valid plates        :",
        valid_count
    )

    print(
        "Corrected plates    :",
        corrected_count
    )

    print(
        "Needs review        :",
        review_count
    )

    print(
        "Too short           :",
        short_count
    )

    print(
        "Empty               :",
        empty_count
    )

    print()
    print("Output file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()