import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions


#loding ai model
model = EfficientNetB3(weights='imagenet')

def recognize_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and preprocess the image
    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((300, 300))


    # Display the image and predictions
    display_img = img.resize((320, 320))
    tk_img = ImageTk.PhotoImage(display_img)

    image_label.config(image=tk_img)
    image_label.image = tk_img


    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=5)[0]

    results_text = "AI Predictions:\n"

    for _, label, confidence in decoded:

        confidence = confidence * 100

        if confidence > 5:
            if confidence > 80:
                resonaing = "High confidence"
            elif confidence > 50:
                resonaing = "Moderate confidence"
            elif confidence > 25:
                resonaing = "Possible confidence"
            else:
                resonaing = "Low confidence"

            results_text += f"{label}: {confidence:.2f}% - {resonaing}\n"
            results_text += f"{label}: {confidence:.2f}%\n"
            results_text += f"reasoning: {resonaing}\n\n"
    results_label.config(text=results_text)

#user interface
root = tk.Tk()
root.title("Image Recognition with EfficientNetB3")
root.geometry("600x700")

title_label = tk.Label(root, text="Image Recognition with EfficientNetB3", font=("Helvetica", 16))
title_label.pack(pady=10)
upload_button = tk.Button(root, text="Upload Image", command=recognize_image)
upload_button.pack(pady=10) 
image_label = tk.Label(root)
image_label.pack()
results_label = tk.Label(root, text="AI Predictions:\n", font=("Helvetica", 12), justify="left")
results_label.pack(pady=20)

root.mainloop()
