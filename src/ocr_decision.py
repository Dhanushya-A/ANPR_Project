import os
import csv

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"D:\ANPR_Project\outputs\ocr_results\validated_ocr_results.csv"

OUTPUT_FILE = r"D:\ANPR_Project\outputs\ocr_results\final_ocr_results.csv"

# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

HIGH_CONFIDENCE = 70
MEDIUM_CONFIDENCE = 50


# ============================================================
# DECISION FUNCTION
# ============================================================

def make_decision(status, confidence):

    try:
        confidence = float(confidence)
    except:
        confidence = 0.0

    # --------------------------------------------------------
    # Definitely unusable
    # --------------------------------------------------------

    if status in ["EMPTY", "TOO_SHORT"]:
        return "REJECT"

    # --------------------------------------------------------
    # Structurally valid
    # --------------------------------------------------------

    if status == "VALID":

        if confidence >= HIGH_CONFIDENCE:
            return "ACCEPT"

        return "REVIEW"

    # --------------------------------------------------------
    # Corrected OCR
    # --------------------------------------------------------

    if status == "CORRECTED":

        if confidence >= HIGH_CONFIDENCE:
            return "ACCEPT"

        return "REVIEW"

    # --------------------------------------------------------
    # Unknown state / review
    # --------------------------------------------------------

    return "REVIEW"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANPR - OCR FINAL DECISION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        print()
        print("ERROR: Input file not found:")
        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Read input
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    print()
    print("Records found:", len(rows))

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    accepted = 0
    review = 0
    rejected = 0

    output_rows = []

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for row in rows:

        image = row.get(
            "image",
            ""
        )

        ocr_text = row.get(
            "ocr_text",
            ""
        )

        validated_text = row.get(
            "validated_text",
            ""
        )

        status = row.get(
            "status",
            ""
        )

        confidence = row.get(
            "confidence",
            "0"
        )

        confidence_level = row.get(
            "confidence_level",
            ""
        )

        decision = make_decision(
            status,
            confidence
        )

        # ----------------------------------------------------
        # Count
        # ----------------------------------------------------

        if decision == "ACCEPT":
            accepted += 1

        elif decision == "REVIEW":
            review += 1

        elif decision == "REJECT":
            rejected += 1

        # ----------------------------------------------------
        # Final plate
        # ----------------------------------------------------

        if decision == "ACCEPT":
            final_plate = validated_text
        else:
            final_plate = ""

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_rows.append(
            {
                "image": image,
                "ocr_text": ocr_text,
                "validated_text": validated_text,
                "final_plate": final_plate,
                "status": status,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "decision": decision
            }
        )

    # --------------------------------------------------------
    # Write CSV
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
            "final_plate",
            "status",
            "confidence",
            "confidence_level",
            "decision"
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
    print("FINAL OCR DECISION COMPLETED")
    print("=" * 70)

    print()
    print("Total records :", len(rows))
    print("Accepted      :", accepted)
    print("Review        :", review)
    print("Rejected      :", rejected)

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()