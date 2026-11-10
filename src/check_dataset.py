from pathlib import Path

# Change this path if your extracted dataset folder has a different name
DATASET_DIR = Path("D:/ANPR_Project/dataset/Indian License Plates with Labels")

images_dir = DATASET_DIR / "images"
labels_dir = DATASET_DIR / "labels"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in images_dir.iterdir()
    if p.suffix.lower() in image_extensions
]

labels = list(labels_dir.glob("*.txt"))

image_names = {p.stem for p in images}
label_names = {p.stem for p in labels}

missing_labels = sorted(image_names - label_names)
extra_labels = sorted(label_names - image_names)

print("=" * 50)
print("ANPR DATASET CHECK")
print("=" * 50)

print(f"Images found       : {len(images)}")
print(f"Labels found       : {len(labels)}")
print(f"Images with labels : {len(image_names & label_names)}")
print(f"Missing labels     : {len(missing_labels)}")
print(f"Extra labels       : {len(extra_labels)}")

print("\nChecking YOLO labels...")

invalid_labels = []

for label_file in labels:
    try:
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()

            if len(parts) != 5:
                invalid_labels.append(
                    f"{label_file.name} - line {line_number}"
                )
                continue

            class_id, x, y, width, height = parts

            # Class must be an integer
            int(class_id)

            values = [float(x), float(y), float(width), float(height)]

            # YOLO normalized coordinates should be between 0 and 1
            if not all(0 <= value <= 1 for value in values):
                invalid_labels.append(
                    f"{label_file.name} - line {line_number}"
                )

    except Exception:
        invalid_labels.append(label_file.name)

print(f"Invalid labels     : {len(invalid_labels)}")

print("\n" + "=" * 50)
print("CHECK COMPLETE")
print("=" * 50)

if missing_labels:
    print("\nFirst 10 images without labels:")
    for name in missing_labels[:10]:
        print(" -", name)

if invalid_labels:
    print("\nFirst 10 invalid labels:")
    for name in invalid_labels[:10]:
        print(" -", name)

if not missing_labels and not invalid_labels:
    print("\nEverything looks good!")

print("\nNext step: create the train/validation/test split.")