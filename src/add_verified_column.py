import pandas as pd
import os

CSV_PATH = r"D:\ANPR_Project\outputs\ocr_review\ocr_review.csv"

print("=" * 60)
print("ADD VERIFIED PLATE COLUMN")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    print("ERROR: OCR review CSV not found.")
    print(CSV_PATH)
    raise SystemExit

df = pd.read_csv(CSV_PATH)

# Add verified_plate column if it does not exist
if "verified_plate" not in df.columns:
    df["verified_plate"] = ""
    print("verified_plate column added.")
else:
    print("verified_plate column already exists.")

# Make sure notes column also exists
if "notes" not in df.columns:
    df["notes"] = ""

df.to_csv(CSV_PATH, index=False)

print()
print("Total records:", len(df))
print("CSV updated successfully:")
print(CSV_PATH)

print()
print("Columns:")
print(list(df.columns))