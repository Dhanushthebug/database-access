import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# ========== Constants ==========
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 6
BATCH_SIZE = 16
EPOCHS = 50

# ========== U-Net++ Decoder ==========
def unetpp_decoder(encoder_output, num_classes):
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(encoder_output)
    x = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(x)
    return x

# ========== Build Model ==========
def build_model(input_shape=(224, 224, 3), num_classes=6):
    inputs = layers.Input(shape=input_shape)
    encoder = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    encoder_output = encoder.output
    decoder_output = unetpp_decoder(encoder_output, num_classes)
    decoder_output = layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(decoder_output)
    return models.Model(inputs, decoder_output)

# ========== Dice Loss ==========
def dice_loss(y_true, y_pred, smooth=1e-7):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# ========== Load Dataset ==========
def load_data(image_dir, mask_dir):
    images, masks = [], []
    filenames = []

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)  # assumes same name

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {fname} due to read error.")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        images.append(image)
        masks.append(mask)
        filenames.append(fname)

    return np.array(images), np.array(masks), filenames

# ========== Load & Normalize ==========
image_dir = r"C:\Users\Dhanush\Desktop\segmentation_data\images"
mask_dir = r"C:\Users\Dhanush\Desktop\segmentation_data\masks"

x_data, y_data, filenames = load_data(image_dir, mask_dir)
x_data = x_data.astype('float32') / 255.0
y_data = y_data.astype('int32')

# ========== Build & Train ==========
model = build_model()
model.compile(optimizer='adam',
              loss=dice_loss,
              metrics=['sparse_categorical_accuracy'])

model.fit(x_data, y_data, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

# ========== Predict All ==========
preds = model.predict(x_data)
preds_mask = np.argmax(preds, axis=-1)

# ========== Evaluate ==========
def calculate_dice_per_class(pred_mask, true_mask, num_classes=6):
    scores = []
    for i in range(num_classes):
        pred_i = (pred_mask == i).astype(np.float32)
        true_i = (true_mask == i).astype(np.float32)
        intersection = np.sum(pred_i * true_i)
        dice = (2. * intersection + 1e-7) / (np.sum(pred_i) + np.sum(true_i) + 1e-7)
        scores.append(dice)
    return scores

# Average Dice per class
dice_scores_all = []
y_true_flat, y_pred_flat = [], []

for i in range(len(filenames)):
    dice_scores = calculate_dice_per_class(preds_mask[i], y_data[i], NUM_CLASSES)
    dice_scores_all.append(dice_scores)
    y_true_flat.extend(y_data[i].flatten())
    y_pred_flat.extend(preds_mask[i].flatten())

# ========== Reporting ==========
avg_dice_scores = np.mean(dice_scores_all, axis=0)
print("Average Dice Scores per Class:")
for idx, score in enumerate(avg_dice_scores):
    print(f"Class {idx}: {score:.4f}")

print("\nClassification Report:")
print(classification_report(y_true_flat, y_pred_flat))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_flat, y_pred_flat))
