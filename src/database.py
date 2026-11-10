import os
import pandas as pd


PROJECT_DIR = r"D:\ANPR_Project"

DATABASE_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "vehicles.csv"
)


def load_database():
    if not os.path.exists(DATABASE_FILE):
        print("ERROR: Database not found:")
        print(DATABASE_FILE)
        return pd.DataFrame()

    return pd.read_csv(DATABASE_FILE)


def normalize_plate(plate):
    return (
        str(plate)
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .strip()
    )


def check_vehicle(plate_number):

    df = load_database()

    if df.empty:
        return False

    plate = normalize_plate(plate_number)

    df["normalized_plate"] = (
        df["plate_number"]
        .apply(normalize_plate)
    )

    result = df[
        (df["normalized_plate"] == plate) &
        (df["authorized"].astype(str).str.upper() == "YES") &
        (df["status"].astype(str).str.upper() == "ACTIVE")
    ]

    return not result.empty


if __name__ == "__main__":

    print("=" * 60)
    print("ANPR VEHICLE DATABASE TEST")
    print("=" * 60)

    test_plates = [
        "MH05AS9787",
        "MH04DW7092",
        "AP10AP9486",
        "UNKNOWN123"
    ]

    for plate in test_plates:

        authorized = check_vehicle(plate)

        if authorized:
            print(f"{plate} -> AUTHORIZED")
        else:
            print(f"{plate} -> NOT AUTHORIZED")