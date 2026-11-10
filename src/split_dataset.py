from pathlib import Path
import random
import shutil

# Dataset location
SOURCE_DIR = Path(
    "D:/ANPR_Project/dataset/Indian License Plates with Labels"
)

IMAGES_DIR = SOURCE_DIR / "images"
LABELS_DIR = SOURCE_DIR / "labels"

# New YOLO dataset location
OUTPUT_DIR = Path("D:/ANPR_Project/dataset/anpr_yolo")

# Number of images for each split
TRAIN_COUNT = 700
VAL_COUNT = 150
TEST_COUNT = 150

# Fixed seed = same split every time
random.seed(42)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Find images that have corresponding labels
valid_images = []

for image_path in IMAGES_DIR.iterdir():

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        valid_images.append(image_path)

print("=" * 50)
print("ANPR DATASET SPLITTER")
print("=" * 50)

print(f"Images with labels found: {len(valid_images)}")

required = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

if len(valid_images) < required:
    raise ValueError(
        f"Not enough labelled images. "
        f"Required: {required}, Found: {len(valid_images)}"
    )

# Shuffle
random.shuffle(valid_images)

# Select images
train_images = valid_images[:TRAIN_COUNT]

val_images = valid_images[
    TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
]

test_images = valid_images[
    TRAIN_COUNT + VAL_COUNT:
    TRAIN_COUNT + VAL_COUNT + TEST_COUNT
]

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images,
}

# Create directories
for split in splits:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# Copy images and labels
for split, image_list in splits.items():

    for image_path in image_list:

        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        destination_image = (
            OUTPUT_DIR / "images" / split / image_path.name
        )

        destination_label = (
            OUTPUT_DIR / "labels" / split / label_path.name
        )

        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)

    print(
        f"{split.upper():5} : {len(image_list)} images"
    )

print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print("\nDataset structure:")
print("anpr_yolo/")
print("├── images/")
print("│   ├── train/  (700)")
print("│   ├── val/    (150)")
print("│   └── test/   (150)")
print("│")
print("└── labels/")
print("    ├── train/  (700)")
print("    ├── val/    (150)")
print("    └── test/   (150)")