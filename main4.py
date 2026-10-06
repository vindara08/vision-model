from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = YOLO("best.pt")


# ==========================================
# 2. COLOR DEFINITIONS
# ==========================================

COLOR_PALETTE = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Light Gray": (190, 190, 190),
    "Gray": (128, 128, 128),
    "Dark Gray": (70, 70, 70),

    "Red": (220, 40, 40),
    "Dark Red": (120, 20, 20),

    "Orange": (240, 130, 30),
    "Yellow": (240, 210, 30),

    "Green": (40, 160, 70),
    "Dark Green": (20, 90, 40),

    "Cyan": (30, 190, 200),

    "Sky Blue": (80, 170, 230),
    "Blue": (40, 90, 200),
    "Navy Blue": (20, 35, 100),

    "Purple": (130, 60, 180),
    "Pink": (230, 100, 160),

    "Brown": (120, 70, 35),
    "Dark Brown": (70, 40, 20),

    "Beige": (210, 190, 150),
}


# ==========================================
# 3. RGB → COLOR NAME
# ==========================================

def closest_color(rgb):

    rgb = np.array(rgb, dtype=np.float32)

    best_name = None
    best_distance = float("inf")

    for name, palette_rgb in COLOR_PALETTE.items():

        palette_rgb = np.array(
            palette_rgb,
            dtype=np.float32
        )

        distance = np.linalg.norm(
            rgb - palette_rgb
        )

        if distance < best_distance:

            best_distance = distance
            best_name = name

    return best_name


# ==========================================
# 4. COLOR FAMILY
# ==========================================

def get_color_family(rgb):

    r, g, b = np.array(rgb) / 255.0

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    h = h * 360
    s = s * 100
    v = v * 100

    # Neutral colors
    if v < 15:
        return "Black"

    if s < 12 and v > 85:
        return "White"

    if s < 15:
        return "Gray"

    # Color families
    if h < 15 or h >= 345:
        return "Red"

    if h < 45:
        return "Orange"

    if h < 70:
        return "Yellow"

    if h < 165:
        return "Green"

    if h < 195:
        return "Cyan"

    if h < 255:
        return "Blue"

    if h < 290:
        return "Purple"

    if h < 345:
        return "Pink"

    return "Unknown"


# ==========================================
# 5. BRIGHTNESS
# ==========================================

def get_brightness(rgb):

    r, g, b = rgb

    brightness = (
        0.299 * r +
        0.587 * g +
        0.114 * b
    )

    if brightness < 70:
        return "Dark"

    elif brightness < 170:
        return "Medium"

    else:
        return "Light"


# ==========================================
# 6. LOAD IMAGE
# ==========================================

image_path = "TheShirtCoreShoot_1_TheFlutterShoulder_White_02063.webp"  # Replace with your image path

image = cv2.imread(image_path)

if image is None:
    print("Could not load image.")
    exit()

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# 7. RUN YOLO
# ==========================================

results = model(
    image_path,
    imgsz=416,
    conf=0.5
)

result = results[0]


# ==========================================
# 8. CHECK DETECTIONS
# ==========================================

if result.masks is None:

    print("No clothing detected.")
    exit()


# ==========================================
# 9. PROCESS EACH CLOTHING ITEM
# ==========================================

for i, mask in enumerate(result.masks.data):

    # ------------------------------
    # Clothing information
    # ------------------------------

    class_id = int(
        result.boxes.cls[i]
    )

    confidence = float(
        result.boxes.conf[i]
    )

    class_name = result.names[class_id]


    # ------------------------------
    # Convert mask
    # ------------------------------

    mask = mask.cpu().numpy()

    mask = cv2.resize(
        mask,
        (
            image_rgb.shape[1],
            image_rgb.shape[0]
        )
    )

    mask = mask > 0.5


    # ------------------------------
    # Extract clothing pixels
    # ------------------------------

    clothing_pixels = image_rgb[mask]

    if len(clothing_pixels) == 0:
        continue


    # ------------------------------
    # Limit pixels
    # ------------------------------

    if len(clothing_pixels) > 5000:

        indices = np.random.choice(
            len(clothing_pixels),
            5000,
            replace=False
        )

        clothing_pixels = (
            clothing_pixels[indices]
        )


    # ==================================
    # 10. KMEANS COLOR EXTRACTION
    # ==================================

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        clothing_pixels
    )

    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)

    percentages = (
        counts /
        counts.sum()
    ) * 100


    # Sort colors by percentage
    sorted_indices = np.argsort(
        percentages
    )[::-1]


    # ==================================
    # 11. DOMINANT COLOR
    # ==================================

    dominant_index = sorted_indices[0]

    dominant_rgb = centers[
        dominant_index
    ].astype(int)

    dominant_percentage = percentages[
        dominant_index
    ]

    dominant_name = closest_color(
        dominant_rgb
    )

    dominant_family = get_color_family(
        dominant_rgb
    )

    dominant_brightness = get_brightness(
        dominant_rgb
    )


    # ==================================
    # 12. SECONDARY COLOR
    # ==================================

    secondary_name = None
    secondary_percentage = 0

    if len(sorted_indices) > 1:

        secondary_index = sorted_indices[1]

        secondary_rgb = centers[
            secondary_index
        ].astype(int)

        secondary_percentage = percentages[
            secondary_index
        ]

        # Only count it as a meaningful
        # secondary color if it occupies
        # enough of the clothing.

        if secondary_percentage >= 10:

            secondary_name = closest_color(
                secondary_rgb
            )


    # ==================================
    # 13. PRINT RESULT
    # ==================================

    print("\n")
    print("=" * 45)

    print(
        f"Clothing: {class_name}"
    )

    print(
        f"Detection confidence: "
        f"{confidence:.2f}"
    )

    print(
        f"Dominant color: "
        f"{dominant_name}"
    )

    print(
        f"Color family: "
        f"{dominant_family}"
    )

    print(
        f"Brightness: "
        f"{dominant_brightness}"
    )

    print(
        f"Dominant RGB: "
        f"{tuple(dominant_rgb)}"
    )

    print(
        f"Dominant percentage: "
        f"{dominant_percentage:.1f}%"
    )

    if secondary_name:

        print(
            f"Secondary color: "
            f"{secondary_name}"
        )

        print(
            f"Secondary percentage: "
            f"{secondary_percentage:.1f}%"
        )

    else:

        print(
            "Secondary color: None"
        )

    print("=" * 45)