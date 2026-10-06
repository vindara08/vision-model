from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


# ==========================================
# 1. LOAD MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO("best.pt")

print("Model loaded successfully.")


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
# 3. PATTERN DETECTION
# ==========================================

def detect_pattern(image_rgb, mask):

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return "unknown", 0.0

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = image_rgb[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    if crop.size == 0:
        return "unknown", 0.0

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    pixels = gray[crop_mask]

    if len(pixels) < 100:
        return "unknown", 0.0

    # -----------------------------------
    # Texture measurement
    # -----------------------------------

    texture_variation = np.std(pixels)

    # -----------------------------------
    # Edge analysis
    # -----------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    masked_edges = edges[crop_mask]

    edge_density = np.mean(
        masked_edges > 0
    )

    # -----------------------------------
    # Horizontal / vertical structure
    # -----------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 15)
    )

    horizontal_lines = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_lines = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    horizontal_strength = np.sum(
        horizontal_lines[crop_mask]
    )

    vertical_strength = np.sum(
        vertical_lines[crop_mask]
    )

    # -----------------------------------
    # Pattern estimation
    # -----------------------------------

    if (
        texture_variation < 18
        and edge_density < 0.08
    ):
        return "solid", 0.80

    if (
        horizontal_strength > 1000
        and vertical_strength > 1000
    ):
        return "checked", 0.65

    if (
        horizontal_strength > 1200
        and horizontal_strength >
        vertical_strength * 1.5
    ):
        return "striped", 0.65

    if (
        vertical_strength > 1200
        and vertical_strength >
        horizontal_strength * 1.5
    ):
        return "striped", 0.60

    if (
        texture_variation > 35
        or edge_density > 0.18
    ):
        return "printed", 0.50

    return "unknown", 0.35


# ==========================================
# 4. RGB → COLOR NAME
# ==========================================

def closest_color(rgb):

    rgb = np.array(
        rgb,
        dtype=np.float32
    )

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
# 5. COLOR FAMILY
# ==========================================

def get_color_family(rgb):

    r, g, b = np.array(rgb) / 255.0

    h, s, v = colorsys.rgb_to_hsv(
        r,
        g,
        b
    )

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
# 6. BRIGHTNESS
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
# 7. ANALYZE IMAGE
# ==========================================

def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None, "Could not load image."

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------
    # Run YOLO
    # --------------------------------------

    results = model(
        image_path,
        imgsz=416,
        conf=0.5
    )

    result = results[0]

    print("\nRAW YOLO DETECTIONS")
    print("=" * 40)

    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i])
        confidence = float(result.boxes.conf[i])

        class_name = result.names[class_id]

        box = result.boxes.xyxy[i].cpu().numpy()

        print(
            f"{i}: "
            f"{class_name} | "
            f"confidence={confidence:.2f} | "
            f"box={box.astype(int)}"
        )

    print("=" * 40)

    # --------------------------------------
    # Check detections
    # --------------------------------------

    if (
        result.masks is None
        or result.boxes is None
        or len(result.boxes) == 0
    ):
        return [], "No supported clothing detected."

    clothing_results = []

    # --------------------------------------
    # Process every detected item
    # --------------------------------------

    for i, mask in enumerate(
        result.masks.data
    ):

        class_id = int(
            result.boxes.cls[i]
        )

        confidence = float(
            result.boxes.conf[i]
        )

        class_name = result.names[
            class_id
        ]

        # ----------------------------------
        # Convert mask
        # ----------------------------------

        mask = mask.cpu().numpy()

        mask = cv2.resize(
            mask,
            (
                image_rgb.shape[1],
                image_rgb.shape[0]
            )
        )

        mask = mask > 0.5

        # ----------------------------------
        # Pattern
        # ----------------------------------

        pattern, pattern_confidence = (
            detect_pattern(
                image_rgb,
                mask
            )
        )

        # ----------------------------------
        # Clothing pixels
        # ----------------------------------

        clothing_pixels = image_rgb[
            mask
        ]

        if len(clothing_pixels) == 0:
            continue

        # ----------------------------------
        # Limit pixels
        # ----------------------------------

        if len(clothing_pixels) > 5000:

            indices = np.random.choice(
                len(clothing_pixels),
                5000,
                replace=False
            )

            clothing_pixels = (
                clothing_pixels[indices]
            )

        # ----------------------------------
        # KMeans
        # ----------------------------------

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(
            clothing_pixels
        )

        centers = kmeans.cluster_centers_

        counts = np.bincount(
            labels
        )

        percentages = (
            counts /
            counts.sum()
        ) * 100

        sorted_indices = np.argsort(
            percentages
        )[::-1]

        # ----------------------------------
        # Dominant color
        # ----------------------------------

        dominant_index = sorted_indices[0]

        dominant_rgb = centers[
            dominant_index
        ].astype(int)

        dominant_percentage = (
            percentages[
                dominant_index
            ]
        )

        dominant_name = closest_color(
            dominant_rgb
        )

        dominant_family = get_color_family(
            dominant_rgb
        )

        dominant_brightness = get_brightness(
            dominant_rgb
        )

        # ----------------------------------
        # Secondary color
        # ----------------------------------

        secondary_name = None
        secondary_percentage = 0

        if len(sorted_indices) > 1:

            secondary_index = (
                sorted_indices[1]
            )

            secondary_rgb = centers[
                secondary_index
            ].astype(int)

            secondary_percentage = (
                percentages[
                    secondary_index
                ]
            )

            if secondary_percentage >= 10:

                secondary_name = (
                    closest_color(
                        secondary_rgb
                    )
                )

        # ----------------------------------
        # Save result
        # ----------------------------------

        clothing_results.append({

            "clothing": class_name,

            "confidence": confidence,

            "pattern": pattern,

            "pattern_confidence":
                pattern_confidence,

            "dominant_color":
                dominant_name,

            "color_family":
                dominant_family,

            "brightness":
                dominant_brightness,

            "dominant_rgb":
                tuple(dominant_rgb),

            "dominant_percentage":
                dominant_percentage,

            "secondary_color":
                secondary_name,

            "secondary_percentage":
                secondary_percentage
        })

    return clothing_results, None


# ==========================================
# 8. TKINTER APPLICATION
# ==========================================

class FashionTester:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Digital Wardrobe - AI Fashion Tester"
        )

        self.root.geometry(
            "1100x750"
        )

        self.root.minsize(
            900,
            650
        )

        self.current_image = None

        self.setup_ui()


    # ======================================
    # UI SETUP
    # ======================================

    def setup_ui(self):

        # ----------------------------------
        # Main title
        # ----------------------------------

        title = tk.Label(
            self.root,
            text="Digital Wardrobe AI Tester",
            font=("Arial", 24, "bold")
        )

        title.pack(
            pady=(15, 5)
        )


        subtitle = tk.Label(
            self.root,
            text=(
                "YOLO clothing detection + "
                "color extraction + pattern detection"
            ),
            font=("Arial", 11)
        )

        subtitle.pack(
            pady=(0, 15)
        )


        # ==================================
        # Main container
        # ==================================

        main_frame = tk.Frame(
            self.root
        )

        main_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )


        # ==================================
        # LEFT SIDE - IMAGE
        # ==================================

        left_frame = tk.Frame(
            main_frame,
            width=500
        )

        left_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )


        image_title = tk.Label(
            left_frame,
            text="Image Preview",
            font=("Arial", 15, "bold")
        )

        image_title.pack(
            pady=5
        )


        self.image_label = tk.Label(
            left_frame,
            text="No image selected",
            relief="solid",
            bd=1,
            width=50,
            height=25
        )

        self.image_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )


        # ==================================
        # RIGHT SIDE - RESULTS
        # ==================================

        right_frame = tk.Frame(
            main_frame,
            width=500
        )

        right_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(10, 0)
        )


        result_title = tk.Label(
            right_frame,
            text="AI Results",
            font=("Arial", 15, "bold")
        )

        result_title.pack(
            pady=5
        )


        # ----------------------------------
        # Scrollable results area
        # ----------------------------------

        result_container = tk.Frame(
            right_frame
        )

        result_container.pack(
            fill="both",
            expand=True
        )


        self.result_text = tk.Text(
            result_container,
            wrap="word",
            font=("Consolas", 11),
            state="disabled"
        )

        self.result_text.pack(
            side="left",
            fill="both",
            expand=True
        )


        scrollbar = tk.Scrollbar(
            result_container,
            command=self.result_text.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.result_text.config(
            yscrollcommand=scrollbar.set
        )


        # ==================================
        # MANUAL ATTRIBUTES
        # ==================================

        manual_frame = tk.LabelFrame(
            self.root,
            text="Manual Clothing Information",
            font=("Arial", 11, "bold")
        )

        manual_frame.pack(
            fill="x",
            padx=20,
            pady=10
        )


        # ----------------------------------
        # Formality
        # ----------------------------------

        tk.Label(
            manual_frame,
            text="Formality:"
        ).grid(
            row=0,
            column=0,
            padx=10,
            pady=8
        )

        self.formality_var = tk.StringVar(
            value="Casual / Informal"
        )

        formality_menu = tk.OptionMenu(
            manual_frame,
            self.formality_var,
            "Casual / Informal",
            "Smart Casual",
            "Formal"
        )

        formality_menu.grid(
            row=0,
            column=1,
            padx=10
        )


        # ----------------------------------
        # Season
        # ----------------------------------

        tk.Label(
            manual_frame,
            text="Season:"
        ).grid(
            row=0,
            column=2,
            padx=10
        )

        self.season_var = tk.StringVar(
            value="All Season"
        )

        season_menu = tk.OptionMenu(
            manual_frame,
            self.season_var,
            "Summer",
            "Winter",
            "Monsoon",
            "All Season"
        )

        season_menu.grid(
            row=0,
            column=3,
            padx=10
        )


        # ----------------------------------
        # Occasion
        # ----------------------------------

        tk.Label(
            manual_frame,
            text="Occasion:"
        ).grid(
            row=0,
            column=4,
            padx=10
        )

        self.occasion_var = tk.StringVar(
            value="Daily Wear"
        )

        occasion_menu = tk.OptionMenu(
            manual_frame,
            self.occasion_var,
            "Daily Wear",
            "College",
            "Office",
            "Party",
            "Travel",
            "Sports",
            "Wedding / Traditional",
            "Other"
        )

        occasion_menu.grid(
            row=0,
            column=5,
            padx=10
        )


        # ==================================
        # BUTTONS
        # ==================================

        button_frame = tk.Frame(
            self.root
        )

        button_frame.pack(
            pady=15
        )


        upload_button = tk.Button(
            button_frame,
            text="Upload Image",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=8,
            command=self.upload_image
        )

        upload_button.pack(
            side="left",
            padx=10
        )


        analyze_button = tk.Button(
            button_frame,
            text="Analyze Image",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=8,
            command=self.analyze_current_image
        )

        analyze_button.pack(
            side="left",
            padx=10
        )


        clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 12),
            padx=20,
            pady=8,
            command=self.clear
        )

        clear_button.pack(
            side="left",
            padx=10
        )


        # ==================================
        # STATUS
        # ==================================

        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 11)
        )

        self.status_label.pack(
            pady=(0, 10)
        )


    # ======================================
    # UPLOAD IMAGE
    # ======================================

    def upload_image(self):

        file_path = filedialog.askopenfilename(

            title="Select Clothing Image",

            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.webp *.bmp"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not file_path:
            return

        self.current_image = file_path

        self.display_image(
            file_path
        )

        self.clear_results()

        self.status_label.config(
            text="Image selected. Click Analyze Image."
        )


    # ======================================
    # DISPLAY IMAGE
    # ======================================

    def display_image(self, image_path):

        try:

            image = Image.open(
                image_path
            )

            image.thumbnail(
                (480, 450),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                image
            )

            self.image_label.config(
                image=photo,
                text=""
            )

            # Very important:
            # keep reference to image
            self.image_label.image = photo

        except Exception as e:

            messagebox.showerror(
                "Image Error",
                str(e)
            )


    # ======================================
    # ANALYZE IMAGE
    # ======================================

    def analyze_current_image(self):

        if not self.current_image:

            messagebox.showwarning(
                "No Image",
                "Please upload an image first."
            )

            return


        self.status_label.config(
            text="Running AI analysis..."
        )

        self.root.update()


        try:

            results, error = analyze_image(
                self.current_image
            )

            # --------------------------------
            # No clothing detected
            # --------------------------------

            if error:

                self.show_rejection(
                    error
                )

                self.status_label.config(
                    text="Rejected - no supported clothing detected."
                )

                return


            # --------------------------------
            # No valid results
            # --------------------------------

            if not results:

                self.show_rejection(
                    "No supported clothing detected."
                )

                self.status_label.config(
                    text="Rejected."
                )

                return


            # --------------------------------
            # Show results
            # --------------------------------

            self.show_results(
                results
            )

            self.status_label.config(
                text=(
                    f"Accepted - "
                    f"{len(results)} clothing item(s) detected."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Analysis Error",
                str(e)
            )

            self.status_label.config(
                text="Error during analysis."


            )


    # ======================================
    # SHOW REJECTION
    # ======================================

    def show_rejection(self, reason):

        self.result_text.config(
            state="normal"
        )

        self.result_text.delete(
            "1.0",
            tk.END
        )

        self.result_text.insert(
            tk.END,
            "\n❌ IMAGE REJECTED\n\n",
        )

        self.result_text.insert(
            tk.END,
            f"Reason:\n{reason}\n\n"
        )

        self.result_text.insert(
            tk.END,
            "Please upload a clear image "
            "containing supported clothing."
        )

        self.result_text.config(
            state="disabled"
        )


    # ======================================
    # SHOW RESULTS
    # ======================================

    def show_results(self, results):

        self.result_text.config(
            state="normal"
        )

        self.result_text.delete(
            "1.0",
            tk.END
        )


        self.result_text.insert(
            tk.END,
            "✅ CLOTHING DETECTED\n"
        )

        self.result_text.insert(
            tk.END,
            "=" * 45 + "\n\n"
        )


        # ----------------------------------
        # Manual fields
        # ----------------------------------

        self.result_text.insert(
            tk.END,
            "MANUAL INFORMATION\n"
        )

        self.result_text.insert(
            tk.END,
            "-" * 30 + "\n"
        )

        self.result_text.insert(
            tk.END,
            f"Formality: "
            f"{self.formality_var.get()}\n"
        )

        self.result_text.insert(
            tk.END,
            f"Season: "
            f"{self.season_var.get()}\n"
        )

        self.result_text.insert(
            tk.END,
            f"Occasion: "
            f"{self.occasion_var.get()}\n\n"
        )


        # ----------------------------------
        # AI results
        # ----------------------------------

        for index, item in enumerate(
            results,
            start=1
        ):

            self.result_text.insert(
                tk.END,
                f"ITEM {index}\n"
            )

            self.result_text.insert(
                tk.END,
                "-" * 30 + "\n"
            )

            self.result_text.insert(
                tk.END,
                f"Clothing: "
                f"{item['clothing']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Detection confidence: "
                f"{item['confidence']:.2f}\n\n"
            )


            # Pattern

            self.result_text.insert(
                tk.END,
                f"Pattern: "
                f"{item['pattern']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Pattern confidence: "
                f"{item['pattern_confidence']:.2f}\n\n"
            )


            # Color

            self.result_text.insert(
                tk.END,
                f"Dominant color: "
                f"{item['dominant_color']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Color family: "
                f"{item['color_family']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Brightness: "
                f"{item['brightness']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Dominant RGB: "
                f"{item['dominant_rgb']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Dominant percentage: "
                f"{item['dominant_percentage']:.1f}%\n\n"
            )


            # Secondary color

            if item["secondary_color"]:

                self.result_text.insert(
                    tk.END,
                    f"Secondary color: "
                    f"{item['secondary_color']}\n"
                )

                self.result_text.insert(
                    tk.END,
                    f"Secondary percentage: "
                    f"{item['secondary_percentage']:.1f}%\n"
                )

            else:

                self.result_text.insert(
                    tk.END,
                    "Secondary color: None\n"
                )


            self.result_text.insert(
                tk.END,
                "\n"
            )

            self.result_text.insert(
                tk.END,
                "=" * 45 + "\n\n"
            )


        self.result_text.config(
            state="disabled"
        )


    # ======================================
    # CLEAR RESULTS
    # ======================================

    def clear_results(self):

        self.result_text.config(
            state="normal"
        )

        self.result_text.delete(
            "1.0",
            tk.END
        )

        self.result_text.config(
            state="disabled"
        )


    # ======================================
    # CLEAR EVERYTHING
    # ======================================

    def clear(self):

        self.current_image = None

        self.image_label.config(
            image="",
            text="No image selected"
        )

        self.image_label.image = None

        self.clear_results()

        self.status_label.config(
            text="Ready"
        )


# ==========================================
# 9. START APPLICATION
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FashionTester(
        root
    )
    

    root.mainloop()