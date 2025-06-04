#single_channel

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0

# Constants
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4  # Change to 3 or 4 as per your data
BATCH_SIZE = 16
EPOCHS = 50

# Decoder definition
def simple_decoder(encoder_output, num_classes):
    x = layers.UpSampling2D(size=(2, 2))(encoder_output)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    return x

# Build model with single channel input
def build_model(input_shape=(224, 224, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # Convert 1 channel grayscale to 3 channels by repeating
    x = layers.Concatenate()([inputs, inputs, inputs])  # Shape: (224, 224, 3)

    # Use EfficientNetB0 pretrained on imagenet with 3 channel input
    encoder = EfficientNetB0(include_top=False, input_tensor=x, weights='imagenet')
    encoder_output = encoder.output

    decoder_output = simple_decoder(encoder_output, num_classes)
    model = models.Model(inputs, decoder_output)
    return model

# Dice loss function
def dice_loss(y_true, y_pred, smooth=1e-7):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    return 1 - tf.reduce_mean((2. * intersection + smooth) / (union + smooth))

# Load data - single channel grayscale images and masks
def load_data(image_dir, mask_dir):
    images, masks = [], []
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {fname} due to read error.")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        # Expand dims to add channel axis: (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=-1)
        mask = mask  # mask stays as (H, W)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Preprocess and split data
def preprocess_and_split(image_dir, mask_dir):
    x, y = load_data(image_dir, mask_dir)
    x = x.astype('float32') / 255.0
    y = tf.one_hot(y.astype('int32'), NUM_CLASSES)
    val_split = int(0.8 * len(x))
    return (x[:val_split], y[:val_split]), (x[val_split:], y[val_split:])

# DIoU score for masks (adapted for segmentation masks)
def calculate_diou(pred_mask, true_mask, num_classes=NUM_CLASSES):
    diou_scores = []
    eps = 1e-7

    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).astype(np.uint8)
        true_cls = (true_mask == cls).astype(np.uint8)

        # Compute centroids
        def centroid(mask):
            coords = np.column_stack(np.where(mask > 0))
            if coords.size == 0:
                return None
            return coords.mean(axis=0)

        c_pred = centroid(pred_cls)
        c_true = centroid(true_cls)

        # If either class not present, skip
        if c_pred is None or c_true is None:
            diou_scores.append(np.nan)
            continue

        # Bounding boxes (min row,col and max row,col)
        def bbox(mask):
            coords = np.column_stack(np.where(mask > 0))
            if coords.size == 0:
                return None
            minr, minc = coords.min(axis=0)
            maxr, maxc = coords.max(axis=0)
            return minr, minc, maxr, maxc

        bbox_pred = bbox(pred_cls)
        bbox_true = bbox(true_cls)
        if bbox_pred is None or bbox_true is None:
            diou_scores.append(np.nan)
            continue

        # Intersection bbox
        inter_minr = max(bbox_pred[0], bbox_true[0])
        inter_minc = max(bbox_pred[1], bbox_true[1])
        inter_maxr = min(bbox_pred[2], bbox_true[2])
        inter_maxc = min(bbox_pred[3], bbox_true[3])

        inter_area = max(0, inter_maxr - inter_minr) * max(0, inter_maxc - inter_minc)
        pred_area = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
        true_area = (bbox_true[2] - bbox_true[0]) * (bbox_true[3] - bbox_true[1])
        union_area = pred_area + true_area - inter_area

        # IoU
        iou = inter_area / (union_area + eps)

        # Distance between centroids squared
        dist_centroids = np.sum((c_pred - c_true) ** 2)

        # Diagonal length of smallest enclosing box
        enc_minr = min(bbox_pred[0], bbox_true[0])
        enc_minc = min(bbox_pred[1], bbox_true[1])
        enc_maxr = max(bbox_pred[2], bbox_true[2])
        enc_maxc = max(bbox_pred[3], bbox_true[3])
        diag_len = np.sum((np.array([enc_maxr, enc_maxc]) - np.array([enc_minr, enc_minc])) ** 2) + eps

        diou = iou - (dist_centroids / diag_len)
        diou_scores.append(diou)

    return diou_scores

# Visualize batch predictions: true vs pred masks
def visualize_batch_predictions(images, true_masks, pred_masks, num_samples=3):
    plt.figure(figsize=(num_samples * 5, 10))
    for i in range(num_samples):
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(np.argmax(true_masks[i], axis=-1), cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
        plt.imshow(np.argmax(pred_masks[i], axis=-1), cmap='jet', vmin=0, vmax=NUM_CLASSES - 1)
        plt.title("Pred Mask")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Paths - change accordingly
image_dir = 'C:/Users/Dhanush/Desktop/segmentation_data/images/'
mask_dir = 'C:/Users/Dhanush/Desktop/segmentation_data/masks/'

# Load and preprocess data
(train_x, train_y), (val_x, val_y) = preprocess_and_split(image_dir, mask_dir)

# Build model
model = build_model()
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Callbacks
model_checkpoint = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

# Train
model.fit(train_x, train_y, validation_data=(val_x, val_y),
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[model_checkpoint])

# Predict on validation set for visualization and evaluation
val_preds = model.predict(val_x)

# Visualize some batch predictions
visualize_batch_predictions(val_x, val_y, val_preds, num_samples=3)

# Convert predictions and ground truth to class labels for metrics
val_pred_labels = np.argmax(val_preds, axis=-1)
val_true_labels = np.argmax(val_y, axis=-1)

# Calculate Dice scores on validation set (average per class)
dice_scores_val = []
for i in range(len(val_pred_labels)):
    scores = []
    for cls in range(NUM_CLASSES):
        pred_i = (val_pred_labels[i] == cls).astype(np.float32)
        true_i = (val_true_labels[i] == cls).astype(np.float32)
        intersection = np.sum(pred_i * true_i)
        dice = (2. * intersection + 1e-7) / (np.sum(pred_i) + np.sum(true_i) + 1e-7)
        scores.append(dice)
    dice_scores_val.append(scores)

dice_scores_val = np.array(dice_scores_val)
print("Mean Dice scores per class on validation set:")
for cls in range(NUM_CLASSES):
    print(f"Class {cls}: {dice_scores_val[:, cls].mean():.4f}")

# Compute DIoU scores on validation set (mean per class)
diou_scores_val = []
for i in range(len(val_pred_labels)):
    diou_scores_val.append(calculate_diou(val_pred_labels[i], val_true_labels[i], NUM_CLASSES))

diou_scores_val = np.array(diou_scores_val)
print("\nMean DIoU scores per class on validation set:")
for cls in range(NUM_CLASSES):
    # Skip NaNs in mean calculation
    cls_scores = diou_scores_val[:, cls]
    cls_scores = cls_scores[~np.isnan(cls_scores)]
    if len(cls_scores) > 0:
        print(f"Class {cls}: {cls_scores.mean():.4f}")
    else:
        print(f"Class {cls}: No valid DIoU scores")

# Classification report and confusion matrix for full validation set
print("\nClassification Report (Validation Set):")
print(classification_report(val_true_labels.flatten(), val_pred_labels.flatten()))

print("\nConfusion Matrix (Validation Set):")
print(confusion_matrix(val_true_labels.flatten(), val_pred_labels.flatten()))
