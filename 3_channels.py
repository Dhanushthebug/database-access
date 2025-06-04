# 3-channels

## libraries 
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, callbacks
from scipy.ndimage import label

# ========== Constants ==========
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4  # set 3 or 4 based on your data
BATCH_SIZE = 16
EPOCHS = 50

# ========== Decoder ==========
def simple_decoder(encoder_output, num_classes):
    x = layers.UpSampling2D(size=(2, 2))(encoder_output)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    return x

# ========== Build Model ==========
def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    encoder = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    encoder_output = encoder.output
    decoder_output = simple_decoder(encoder_output, num_classes)
    return models.Model(inputs, decoder_output)

# ========== Dice Loss ==========
def dice_loss(y_true, y_pred, smooth=1e-7):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    return 1 - tf.reduce_mean((2. * intersection + smooth) / (union + smooth))

# ========== Load Data (Option B) ==========
def load_data(image_dir, mask_dir):
    images, masks = [], []
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        # Load grayscale image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {fname} due to read error.")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        # Convert grayscale to 3 channel by repeating same channel 3 times
        image = np.stack([image]*3, axis=-1)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# ========== Prepare Dataset ==========
def preprocess_and_split(image_dir, mask_dir):
    x, y = load_data(image_dir, mask_dir)
    x = x.astype('float32') / 255.0
    y = tf.one_hot(y.astype('int32'), NUM_CLASSES)

    val_split = int(0.8 * len(x))
    return (x[:val_split], y[:val_split]), (x[val_split:], y[val_split:])

# ========== Paths ==========
image_dir = 'C:/Users/Dhanush/Desktop/segmentation_data/images/'
mask_dir = 'C:/Users/Dhanush/Desktop/segmentation_data/masks/'

(train_x, train_y), (val_x, val_y) = preprocess_and_split(image_dir, mask_dir)

# ========== Compile & Train ==========
model = build_model()
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

model_checkpoint = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

model.fit(train_x, train_y, validation_data=(val_x, val_y),
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[model_checkpoint])

# ========== Evaluate Single Sample ==========
def evaluate_single_sample(image_path, mask_path):
    test_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    test_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    test_img = cv2.resize(test_img, IMAGE_SIZE)
    test_img = np.stack([test_img]*3, axis=-1)
    test_img = test_img.astype('float32') / 255.0
    test_mask = cv2.resize(test_mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    test_img_tensor = np.expand_dims(test_img, axis=0)
    test_pred = model.predict(test_img_tensor)
    test_pred_mask = np.argmax(test_pred, axis=-1).squeeze()

    return test_pred_mask, test_mask

# ========== Dice Score Calculation ==========
def calculate_dice_per_class(pred_mask, true_mask, num_classes=NUM_CLASSES):
    scores = []
    for i in range(num_classes):
        pred_i = (pred_mask == i).astype(np.float32)
        true_i = (true_mask == i).astype(np.float32)
        intersection = np.sum(pred_i * true_i)
        dice = (2. * intersection + 1e-7) / (np.sum(pred_i) + np.sum(true_i) + 1e-7)
        scores.append(dice)
    return scores

# ========== DIoU Score Calculation for Masks ==========
def diou_score(mask_pred, mask_true, class_id):
    """
    Calculate DIoU between predicted and true mask for a particular class.
    Steps:
    - Find bounding boxes of connected components in predicted and true masks for class_id
    - Compute DIoU between the two boxes
    
    This is a simplified approach:
    - Find largest connected component in pred and true mask for the class
    - Compute DIoU between those two boxes
    
    Returns DIoU score (0 to 1, higher better)
    """

    def get_bbox(mask):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (x_min, y_min, x_max, y_max)  # (xmin, ymin, xmax, ymax)

    # Get binary masks for class_id
    pred_bin = (mask_pred == class_id).astype(np.uint8)
    true_bin = (mask_true == class_id).astype(np.uint8)

    # Get largest connected component bbox in predicted mask
    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return 0.0  # No overlap possible

    labeled_pred, n_pred = label(pred_bin)
    labeled_true, n_true = label(true_bin)

    def largest_bbox(labeled_mask, n_labels):
        max_area = 0
        bbox = None
        for i in range(1, n_labels+1):
            comp = (labeled_mask == i)
            coords = np.argwhere(comp)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                bbox = (x_min, y_min, x_max, y_max)
        return bbox

    bbox_pred = largest_bbox(labeled_pred, n_pred)
    bbox_true = largest_bbox(labeled_true, n_true)

    if bbox_pred is None or bbox_true is None:
        return 0.0

    # Calculate IoU
    x_min_inter = max(bbox_pred[0], bbox_true[0])
    y_min_inter = max(bbox_pred[1], bbox_true[1])
    x_max_inter = min(bbox_pred[2], bbox_true[2])
    y_max_inter = min(bbox_pred[3], bbox_true[3])

    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    area_pred = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
    area_true = (bbox_true[2] - bbox_true[0]) * (bbox_true[3] - bbox_true[1])

    union_area = area_pred + area_true - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    # Calculate center distance squared
    cx_pred = (bbox_pred[0] + bbox_pred[2]) / 2
    cy_pred = (bbox_pred[1] + bbox_pred[3]) / 2
    cx_true = (bbox_true[0] + bbox_true[2]) / 2
    cy_true = (bbox_true[1] + bbox_true[3]) / 2

    center_dist = (cx_pred - cx_true)**2 + (cy_pred - cy_true)**2

    # Calculate enclosing box
    x_min_enclose = min(bbox_pred[0], bbox_true[0])
    y_min_enclose = min(bbox_pred[1], bbox_true[1])
    x_max_enclose = max(bbox_pred[2], bbox_true[2])
    y_max_enclose = max(bbox_pred[3], bbox_true[3])

    enclose_diag = (x_max_enclose - x_min_enclose)**2 + (y_max_enclose - y_min_enclose)**2

    diou = iou - (center_dist / enclose_diag) if enclose_diag > 0 else iou

    return max(0, diou)

# ========== Batch Prediction Visualization ==========
def visualize_batch_predictions(images, true_masks, model, num_samples=4):
    """
    Visualize true masks and predicted masks side by side for a batch of images.
    images: numpy array of shape (batch, H, W, 3)
    true_masks: one-hot encoded masks shape (batch, H, W, num_classes)
    """
    preds = model.predict(images[:num_samples])
    pred_masks = np.argmax(preds, axis=-1)
    true_masks_decoded = np.argmax(true_masks[:num_samples], axis=-1)

    plt.figure(figsize=(num_samples * 5, 10))
    for i in range(num_samples):
        # Show input image
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis('off')

        # Show true mask
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(true_masks_decoded[i], cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
        plt.title("True Mask")
        plt.axis('off')

        # Show predicted mask
        plt.subplot(3, num_samples, i + 1 + 2*num_samples)
        plt.imshow(pred_masks[i], cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ========== Run Sample Evaluation ==========
test_image_path = r"C:/Users/Dhanush/Desktop/LRDE INTERNSHIP/main code/segmentation_data/image.png"
test_mask_path = r"C:/Users/Dhanush/Desktop/LRDE INTERNSHIP/main code/segmentation_data/mask/ground_truth_mask.png"

pred_mask, true_mask = evaluate_single_sample(test_image_path, test_mask_path)
dice_scores = calculate_dice_per_class(pred_mask, true_mask)
print("Test Dice Scores:", dice_scores)

print("DIoU Scores per Class:")
for c in range(NUM_CLASSES):
    dscore = diou_score(pred_mask, true_mask, c)
    print(f"Class {c}: DIoU = {dscore:.4f}")

print("Classification Report:")
print(classification_report(true_mask.flatten(), pred_mask.flatten()))

print("Confusion Matrix:")
print(confusion_matrix(true_mask.flatten(), pred_mask.flatten()))

# ========== Visualize Batch ==========
visualize_batch_predictions(val_x, val_y, model, num_samples=4)
