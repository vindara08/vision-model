from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans


# -----------------------------------
# 1. Load YOLO26 DeepFashion2 model
# -----------------------------------

model = YOLO("best.pt")


# -----------------------------------
# 2. Run YOLO on your image
# -----------------------------------

image_path = "20260911_120017.jpg"  # Replace with your image path

results = model(
    image_path,
    imgsz=416,
    conf=0.5
)

result = results[0]


# -----------------------------------
# 3. Check whether anything detected
# -----------------------------------

if result.masks is None:
    print("No clothing detected.")
    exit()


# Original image
image = cv2.imread(image_path)

# Convert BGR → RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -----------------------------------
# 4. Process every detected clothing item
# -----------------------------------

for i, mask in enumerate(result.masks.data):

    # Get class information
    class_id = int(result.boxes.cls[i])
    confidence = float(result.boxes.conf[i])
    class_name = result.names[class_id]

    # Convert mask to NumPy
    mask = mask.cpu().numpy()

    # Resize mask to original image size
    mask = cv2.resize(
        mask,
        (image_rgb.shape[1], image_rgb.shape[0])
    )

    # Convert to binary mask
    mask = mask > 0.5

    # -----------------------------------
    # 5. Extract only clothing pixels
    # -----------------------------------

    clothing_pixels = image_rgb[mask]

    if len(clothing_pixels) == 0:
        continue

    # -----------------------------------
    # 6. Find dominant color
    # -----------------------------------

    # Reduce pixels if there are too many
    if len(clothing_pixels) > 5000:
        indices = np.random.choice(
            len(clothing_pixels),
            5000,
            replace=False
        )
        clothing_pixels = clothing_pixels[indices]

    # KMeans finds major color groups
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(clothing_pixels)

    # Number of pixels belonging to each color
    counts = np.bincount(kmeans.labels_)

    # Most common color
    dominant_color = kmeans.cluster_centers_[
        np.argmax(counts)
    ]

    dominant_color = dominant_color.astype(int)

    # -----------------------------------
    # 7. Print result
    # -----------------------------------

    print("\n-----------------------------")
    print(f"Clothing: {class_name}")
    print(f"Confidence: {confidence:.2f}")

    print(
        f"Dominant RGB: "
        f"{tuple(dominant_color)}"
    )