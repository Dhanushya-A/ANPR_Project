from ultralytics import YOLO
import cv2
import os


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = r"D:\ANPR_Project\runs\plate_detector_20epochs-2\weights\best.pt"

INPUT_DIR = r"D:\ANPR_Project\dataset\anpr_yolo\images\test"

OUTPUT_DETECTED = r"D:\ANPR_Project\outputs\detected"
OUTPUT_PLATES = r"D:\ANPR_Project\outputs\plates"


# Create output folders
os.makedirs(OUTPUT_DETECTED, exist_ok=True)
os.makedirs(OUTPUT_PLATES, exist_ok=True)


# -----------------------------
# Load trained YOLO model
# -----------------------------
model = YOLO(MODEL_PATH)


# -----------------------------
# Process test images
# -----------------------------
image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

print(f"Found {len(image_files)} images.")


for image_name in image_files:

    image_path = os.path.join(INPUT_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_name}")
        continue

    # Run YOLO detection
    results = model.predict(
        source=image,
        imgsz=416,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Copy image for drawing detections
    annotated = image.copy()

    detected_count = 0

    if result.boxes is not None:

        for i, box in enumerate(result.boxes):

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            confidence = float(box.conf[0])

            # Keep coordinates inside image
            h, w = image.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Crop plate
            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            detected_count += 1

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display confidence
            label = f"Plate {confidence:.2f}"

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Save plate crop
            base_name = os.path.splitext(image_name)[0]

            plate_filename = (
                f"{base_name}_plate_{i + 1}.jpg"
            )

            plate_path = os.path.join(
                OUTPUT_PLATES,
                plate_filename
            )

            cv2.imwrite(
                plate_path,
                plate_crop
            )

    # Save annotated vehicle image
    detected_path = os.path.join(
        OUTPUT_DETECTED,
        image_name
    )

    cv2.imwrite(
        detected_path,
        annotated
    )

    print(
        f"{image_name} -> "
        f"{detected_count} plate(s) detected"
    )


print("\nDetection and cropping completed.")
print(f"Detected images: {OUTPUT_DETECTED}")
print(f"Plate crops: {OUTPUT_PLATES}")