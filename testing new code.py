import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Set device to CPU to avoid GPU-related issues
device = torch.device("cpu")

# --------------- FCN-8s ---------------
class FCN8s(nn.Module):
    def __init__(self, num_classes=6):
        super(FCN8s, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.decoder = nn.ConvTranspose2d(128, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

fcn_model = FCN8s().to(device)
fcn_model.eval()
print("✅ FCN-8s Model initialized.")

# --------------- DC-GAN ---------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 8 * 8 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 16, stride=32, padding=7),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

generator = Generator().to(device)
generator.eval()
print("✅ DC-GAN Generator initialized.")

# --------------- Preprocessing ---------------
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, (512, 512))
    image = image / 255.0
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found: {mask_path}")
    return cv2.resize(mask, (512, 512))

# --------------- Load Test Data ---------------
test_image_path = "path/to/test/image.jpg"
ground_truth_mask_path = "path/to/mask.png"

test_image = preprocess_image(test_image_path)
ground_truth_mask = preprocess_mask(ground_truth_mask_path)

# --------------- FCN-8s Prediction ---------------
with torch.no_grad():
    output_fcn = fcn_model(test_image)
    pred_mask_fcn = torch.argmax(output_fcn, dim=1).squeeze().cpu().numpy()

# --------------- DC-GAN Prediction ---------------
with torch.no_grad():
    noise = torch.randn((1, 100)).to(device)
    generated_mask = generator(noise).squeeze().cpu().numpy()
    generated_mask = (generated_mask * 255).astype(np.uint8)

# --------------- Visualization ---------------
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("FCN-8s Prediction")
plt.imshow(pred_mask_fcn, cmap="jet")

plt.subplot(1, 2, 2)
plt.title("DC-GAN Generated Mask")
plt.imshow(generated_mask, cmap="gray")
plt.tight_layout()
plt.show()

# --------------- Classification Report ---------------
ground_truth_flat = ground_truth_mask.flatten()
fcn_flat = pred_mask_fcn.flatten()
report_fcn = classification_report(ground_truth_flat, fcn_flat, digits=6, zero_division=0)

print("Classification Report - FCN-8s:\n", report_fcn)
