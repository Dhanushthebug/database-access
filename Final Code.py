import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import classification_report
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names
NUM_CLASSES = 6
CLASS_LABELS = ['background', 'barren land', 'roads', 'urban', 'vegetation', 'water']

# -------- Dataset Definition --------
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        return image, mask

# Example file paths (replace with actual paths)
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
mask_paths = ["mask1.png", "mask2.png", "mask3.png"]

# Example transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

# Dataset and DataLoader
dataset = SegmentationDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------- Dice Coefficient Function --------
def dice_coefficient(pred, target, smooth=1e-7):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# -------- Model Definitions --------
# U-Net++ with EfficientNet backbone
unet_model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
).to(device)

# FCN-8s Model (as defined earlier)
class FCN8s(nn.Module):
    def __init__(self):
        super(FCN8s, self).__init__()
        # Define FCN-8s layers here...

    def forward(self, x):
        # Define forward pass for FCN-8s here...
        return x

fcn_model = FCN8s().to(device)

# DC-GAN Generator (as defined earlier)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define DC-GAN Generator layers here...

    def forward(self, z):
        # Define forward pass for DC-GAN Generator
        return z

generator = Generator().to(device)
discriminator = Discriminator().to(device)  # DC-GAN Discriminator (defined earlier)

# -------- Training Setup --------
criterion = torch.nn.CrossEntropyLoss()
optimizer_unet = torch.optim.Adam(unet_model.parameters(), lr=1e-4)
optimizer_fcn = torch.optim.Adam(fcn_model.parameters(), lr=1e-4)

# Optimizers for DC-GAN
optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# -------- Training Loop for U-Net++ and FCN-8s with Dice --------
epochs = 10  # Define number of epochs

def calculate_dice(pred_mask, true_mask, num_classes):
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()
        dice = dice_coefficient(pred_cls, true_cls)
        dice_scores.append(dice.item())
    return dice_scores

for epoch in range(epochs):
    unet_model.train()
    fcn_model.train()
    
    total_dice_unet = np.zeros(NUM_CLASSES)
    total_dice_fcn = np.zeros(NUM_CLASSES)
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # U-Net++ Training
        optimizer_unet.zero_grad()
        outputs_unet = unet_model(images)
        loss_unet = criterion(outputs_unet, masks)
        loss_unet.backward()
        optimizer_unet.step()

        # Calculate Dice for U-Net++
        pred_mask_unet = torch.argmax(outputs_unet, dim=1)
        dice_unet = calculate_dice(pred_mask_unet, masks, NUM_CLASSES)
        total_dice_unet += np.array(dice_unet)

        # FCN-8s Training
        optimizer_fcn.zero_grad()
        outputs_fcn = fcn_model(images)
        loss_fcn = criterion(outputs_fcn, masks)
        loss_fcn.backward()
        optimizer_fcn.step()

        # Calculate Dice for FCN-8s
        pred_mask_fcn = torch.argmax(outputs_fcn, dim=1)
        dice_fcn = calculate_dice(pred_mask_fcn, masks, NUM_CLASSES)
        total_dice_fcn += np.array(dice_fcn)

    avg_dice_unet = total_dice_unet / len(dataloader)
    avg_dice_fcn = total_dice_fcn / len(dataloader)
    
    print(f"Epoch [{epoch+1}/{epochs}], U-Net++ Loss: {loss_unet.item()}, FCN-8s Loss: {loss_fcn.item()}")
    print(f"U-Net++ Average Dice: {avg_dice_unet}")
    print(f"FCN-8s Average Dice: {avg_dice_fcn}")

# -------- Model Evaluation --------
with torch.no_grad():
    test_image = preprocess_image("test_image.jpg")
    output_unet = unet_model(test_image)
    pred_mask_unet = torch.argmax(output_unet, dim=1).squeeze().cpu().numpy()

    output_fcn = fcn_model(test_image)
    pred_mask_fcn = torch.argmax(output_fcn, dim=1).squeeze().cpu().numpy()

    # Generate synthetic mask using DC-GAN
    noise = torch.randn((1, 100)).to(device)
    generated_mask = generator(noise).squeeze().cpu().numpy()
    generated_mask = (generated_mask * 255).astype(np.uint8)

    # Calculate Dice score on the test set
    ground_truth_mask = cv2.imread("test_mask.png", cv2.IMREAD_GRAYSCALE)  # Replace with actual mask
    dice_unet_test = calculate_dice(pred_mask_unet, ground_truth_mask, NUM_CLASSES)
    dice_fcn_test = calculate_dice(pred_mask_fcn, ground_truth_mask, NUM_CLASSES)

    print("U-Net++ Test Dice Scores:", dice_unet_test)
    print("FCN-8s Test Dice Scores:", dice_fcn_test)

# -------- Visualization --------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Ground Truth")
plt.imshow(ground_truth_mask, cmap="jet")

plt.subplot(1, 3, 2)
plt.title("U-Net++ Prediction")
plt.imshow(pred_mask_unet, cmap="jet")

plt.subplot(1, 3, 3)
plt.title("FCN-8s Prediction")
plt.imshow(pred_mask_fcn, cmap="jet")

plt.show()

# -------- Classification Report --------
print("U-Net++ Classification Report:")
print(classification_report(ground_truth_mask.flatten(), pred_mask_unet.flatten(), target_names=CLASS_LABELS))

print("FCN-8s Classification Report:")
print(classification_report(ground_truth_mask.flatten(), pred_mask_fcn.flatten(), target_names=CLASS_LABELS))
