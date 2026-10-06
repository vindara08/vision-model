from ultralytics import YOLO

# Load HasanKabir23's DeepFashion2 model
model = YOLO("best.pt")

# Run inference
results = model("TheShirtCoreShoot_1_TheFlutterShoulder_White_02063.webp", imgsz=416, conf=0.5)

# Display result
results[0].show()

# Print detected classes
for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            print(
                f"Detected: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )