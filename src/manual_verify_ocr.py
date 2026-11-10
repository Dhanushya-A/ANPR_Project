import os
import sys
import time
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CSV_PATH = r"D:\ANPR_Project\outputs\ocr_review\ocr_review.csv"
IMAGE_DIR = r"D:\ANPR_Project\outputs\ocr_review"


# ============================================================
# NORMALIZE PLATE TEXT
# ============================================================

def normalize_plate(text):
    """
    Normalize plate text for comparison.

    Examples:
        MH05-AS-9787 -> MH05AS9787
        mh05 as 9787 -> MH05AS9787
    """

    if text is None:
        return ""

    text = str(text).strip().upper()

    text = "".join(
        character
        for character in text
        if character.isalnum()
    )

    return text


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(filename):
    """
    Find the plate image in the OCR review folder.
    """

    possible_paths = [
        os.path.join(IMAGE_DIR, filename),
        os.path.join(IMAGE_DIR, "plates", filename),
    ]

    for path in possible_paths:

        if os.path.exists(path):
            return path

    return None


# ============================================================
# OPEN IMAGE WITH WINDOWS
# ============================================================

def open_image_windows(image_path):
    """
    Open image using the Windows default image viewer.

    This avoids the OpenCV gray-window problem.
    """

    try:

        os.startfile(image_path)

        return True

    except Exception as error:

        print()
        print("ERROR: Could not open image with Windows.")
        print(error)

        return False


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(df):

    try:

        df.to_csv(
            CSV_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        return True

    except Exception as error:

        print()
        print("ERROR: Could not save CSV.")
        print(error)

        return False


# ============================================================
# UPDATE VERIFICATION
# ============================================================

def update_verification(
    df,
    index,
    verified_text,
    note=""
):
    """
    Update verified plate and correctness.
    """

    recommended = normalize_plate(
        df.loc[index, "recommended_plate"]
    )

    verified = normalize_plate(
        verified_text
    )

    df.loc[index, "verified_plate"] = verified

    if verified == "":

        df.loc[index, "is_correct"] = ""

        df.loc[index, "notes"] = note

    else:

        if verified == recommended:

            df.loc[index, "is_correct"] = "YES"

        else:

            df.loc[index, "is_correct"] = "NO"

        df.loc[index, "notes"] = note

    return df


# ============================================================
# PRINT RECORD
# ============================================================

def print_record(df, index, total):

    filename = str(
        df.loc[index, "filename"]
    )

    recommended = str(
        df.loc[index, "recommended_plate"]
    )

    confidence = df.loc[index, "confidence"]
    support = df.loc[index, "support_count"]
    structure = df.loc[index, "structure_score"]
    consensus = df.loc[index, "consensus_score"]

    print()
    print("=" * 70)
    print(f"IMAGE {index + 1} / {total}")
    print("=" * 70)

    print(f"Filename           : {filename}")
    print(f"Recommended plate  : {recommended}")
    print(f"OCR confidence     : {confidence}")
    print(f"OCR support        : {support}")
    print(f"Structure score    : {structure}")
    print(f"Consensus score    : {consensus}")

    print()
    print("The plate image has been opened in Windows.")
    print()
    print("Commands:")
    print("  Enter plate number  -> verify")
    print("  S                   -> skip / unreadable")
    print("  R                   -> reset this record")
    print("  Q                   -> quit and save")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANPR MANUAL OCR VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):

        print()
        print("ERROR: OCR review CSV was not found.")
        print(CSV_PATH)

        sys.exit(1)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            CSV_PATH,
            dtype=str,
            keep_default_na=False
        )

    except Exception as error:

        print()
        print("ERROR: Could not read CSV.")
        print(error)

        sys.exit(1)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "filename",
        "recommended_plate",
        "verified_plate",
        "is_correct",
        "notes"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing required columns:")
        print(missing_columns)

        print()
        print(
            "Run add_verified_column.py first."
        )

        sys.exit(1)

    total = len(df)

    if total == 0:

        print()
        print("No OCR review records found.")

        sys.exit(0)

    # --------------------------------------------------------
    # Count current status
    # --------------------------------------------------------

    verified_count = 0
    skipped_count = 0
    remaining_count = 0

    for index in range(total):

        verified = normalize_plate(
            df.loc[index, "verified_plate"]
        )

        notes = str(
            df.loc[index, "notes"]
        ).strip()

        if verified:

            verified_count += 1

        elif notes:

            skipped_count += 1

        else:

            remaining_count += 1

    print()
    print(f"Total review records: {total}")
    print(f"Already verified    : {verified_count}")
    print(f"Already skipped     : {skipped_count}")
    print(f"Remaining           : {remaining_count}")

    print()
    print("Each plate image will open using Windows.")
    print()
    print("IMPORTANT:")
    print(
        "Enter the plate number you can actually "
        "read from the image."
    )

    print(
        "Do NOT simply copy the recommended OCR result."
    )

    print()
    print("If the image is unreadable, enter S.")
    print("If you want to stop, enter Q.")

    print()

    input("Press ENTER to start...")

    # ========================================================
    # PROCESS ALL RECORDS
    # ========================================================

    for index in range(total):

        filename = str(
            df.loc[index, "filename"]
        )

        # ----------------------------------------------------
        # Existing verified record
        # ----------------------------------------------------

        existing_verified = normalize_plate(
            df.loc[index, "verified_plate"]
        )

        if existing_verified:

            print()
            print(
                f"[{index + 1}/{total}] "
                f"Already verified: "
                f"{filename} -> "
                f"{existing_verified}"
            )

            continue

        # ----------------------------------------------------
        # Existing skipped record
        # ----------------------------------------------------

        existing_notes = str(
            df.loc[index, "notes"]
        ).strip()

        if existing_notes:

            print()
            print(
                f"[{index + 1}/{total}] "
                f"Already skipped: "
                f"{filename} -> "
                f"{existing_notes}"
            )

            continue

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print_record(
            df,
            index,
            total
        )

        # ----------------------------------------------------
        # Find image
        # ----------------------------------------------------

        image_path = find_image(filename)

        if image_path is None:

            print()
            print("WARNING: Image not found.")
            print(
                f"Filename: {filename}"
            )

            print()
            print(
                "Enter plate, S to skip, "
                "or Q to quit."
            )

            user_input = input(
                "Verified plate / command: "
            ).strip()

            command = user_input.upper()

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            if command == "Q":

                save_csv(df)

                print()
                print("Progress saved.")
                print("Program stopped.")

                break

            # ------------------------------------------------
            # SKIP
            # ------------------------------------------------

            elif command == "S":

                df = update_verification(
                    df,
                    index,
                    "",
                    "Image not found"
                )

                save_csv(df)

                print(
                    "Marked as skipped."
                )

                continue

            # ------------------------------------------------
            # VERIFY
            # ------------------------------------------------

            else:

                df = update_verification(
                    df,
                    index,
                    user_input
                )

                save_csv(df)

                print()
                print(
                    "Verified plate saved:",
                    df.loc[
                        index,
                        "verified_plate"
                    ]
                )

                continue

        # ----------------------------------------------------
        # Open image using Windows
        # ----------------------------------------------------

        print()
        print("Opening image:")

        print(image_path)

        opened = open_image_windows(
            image_path
        )

        if not opened:

            print()
            print(
                "You can still enter the plate "
                "manually."
            )

        else:

            print()
            print(
                "Image opened in Windows."
            )

            print(
                "Look at the plate carefully."
            )

        # ----------------------------------------------------
        # Small delay to allow Photos to open
        # ----------------------------------------------------

        time.sleep(1)

        # ----------------------------------------------------
        # User input
        # ----------------------------------------------------

        user_input = input(
            "Verified plate / "
            "S=skip / R=reset / Q=quit: "
        ).strip()

        command = user_input.upper()

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if command == "Q":

            save_csv(df)

            print()
            print("=" * 70)
            print("PROGRESS SAVED")
            print("=" * 70)

            print()
            print(
                f"CSV: {CSV_PATH}"
            )

            print()
            print(
                "Run the program again later "
                "to continue."
            )

            break

        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        elif command == "S":

            df = update_verification(
                df,
                index,
                "",
                "Too blurry to verify"
            )

            save_csv(df)

            print()
            print(
                "Skipped."
            )

            print(
                "Reason: Too blurry to verify."
            )

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        elif command == "R":

            df.loc[
                index,
                "verified_plate"
            ] = ""

            df.loc[
                index,
                "is_correct"
            ] = ""

            df.loc[
                index,
                "notes"
            ] = ""

            save_csv(df)

            print()
            print(
                "Record reset."
            )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        else:

            verified_plate = normalize_plate(
                user_input
            )

            if verified_plate == "":

                print()
                print(
                    "No plate entered."
                )

                print(
                    "Use S if the plate "
                    "cannot be read."
                )

                continue

            # ------------------------------------------------
            # Save verification
            # ------------------------------------------------

            df = update_verification(
                df,
                index,
                verified_plate
            )

            save_csv(df)

            recommended = normalize_plate(
                df.loc[
                    index,
                    "recommended_plate"
                ]
            )

            print()
            print("-" * 70)
            print("VERIFICATION SAVED")
            print("-" * 70)

            print(
                f"Recommended : {recommended}"
            )

            print(
                f"Verified    : {verified_plate}"
            )

            if verified_plate == recommended:

                print(
                    "Result      : CORRECT"
                )

            else:

                print(
                    "Result      : INCORRECT"
                )

            print("-" * 70)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    save_csv(df)

    correct_count = 0
    incorrect_count = 0
    skipped_count = 0
    remaining_count = 0

    for index in range(total):

        verified = normalize_plate(
            df.loc[index, "verified_plate"]
        )

        result = str(
            df.loc[index, "is_correct"]
        ).strip().upper()

        notes = str(
            df.loc[index, "notes"]
        ).strip()

        if result == "YES":

            correct_count += 1

        elif result == "NO":

            incorrect_count += 1

        elif (
            not verified
            and notes
        ):

            skipped_count += 1

        else:

            remaining_count += 1

    completed = (
        correct_count
        + incorrect_count
    )

    print()
    print("=" * 70)
    print("OCR VERIFICATION SUMMARY")
    print("=" * 70)

    print(
        f"Total records       : {total}"
    )

    print(
        f"Correct             : "
        f"{correct_count}"
    )

    print(
        f"Incorrect           : "
        f"{incorrect_count}"
    )

    print(
        f"Skipped/unreadable  : "
        f"{skipped_count}"
    )

    print(
        f"Remaining           : "
        f"{remaining_count}"
    )

    if completed > 0:

        accuracy = (
            correct_count
            / completed
        ) * 100

        print()
        print(
            f"OCR accuracy "
            f"(verified readable images): "
            f"{accuracy:.2f}%"
        )

    else:

        print()
        print(
            "OCR accuracy: "
            "Not available yet."
        )

    print()
    print("CSV saved:")
    print(CSV_PATH)

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()