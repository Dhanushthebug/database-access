import cv2
import numpy as np
import os
from skimage import measure
from pathlib import Path

# === CONFIGURATION ===
input_folder = "/mnt/data/input_images"       # Folder containing 40 images
output_root = "/mnt/data/processed_output"    # Root folder for output

# Create output root directory
Path(output_root).mkdir(parents=True, exist_ok=True)

# Supported image extensions
extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# === PROCESS EACH IMAGE ===
for image_name in os.listdir(input_folder):
    if not image_name.lower().endswith(extensions):
        continue  # Skip non-image files

    input_path = os.path.join(input_folder, image_name)
    image_basename = os.path.splitext(image_name)[0]
    output_dir = os.path.join(output_root, image_basename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load grayscale image ---
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Could not read image: {input_path}")
        continue

    # --- Denoising with Gaussian blur ---
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "denoised_image.jpg"), denoised)

    # --- Adaptive thresholding ---
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # --- Morphological noise removal ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite(os.path.join(output_dir, "binary_mask.jpg"), cleaned)

    # --- Label connected components ---
    labels = measure.label(cleaned, connectivity=2)
    label_image = (labels % 256).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "label_image.jpg"), label_image)

    # --- Ground truth mask ---
    ground_truth_mask = (labels > 0).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, "ground_truth_mask.jpg"), ground_truth_mask)

    print(f"✅ Processed: {image_name} → Saved to: {output_dir}")

print("🎉 All images processed successfully!")
