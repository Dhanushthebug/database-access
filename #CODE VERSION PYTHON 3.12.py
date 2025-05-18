#CODE VERSION PYTHON 3.12

##libraries 
tensorflow>=2.15.0
opencv-python>=4.9.0
scikit-learn>=1.3.0
numpy>=1.26.0
##

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

# ========== U-Net++ Decoder (Simplified) ==========
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

# ========== Load Data ==========
def load_data(image_dir, mask_dir):
    images, masks = [], []
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {fname} due to read error.")
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# ========== Dataset Preparation ==========
image_dir = 'C:/Users/Dhanush/Desktop/segmentation_data/images/'
mask_dir = 'C:/Users/Dhanush/Desktop/segmentation_data/masks/'

x_train, y_train = load_data(image_dir, mask_dir)
x_train = x_train.astype('float32') / 255.0
y_train = y_train.astype('int32')

# ========== Compile & Train ==========
model = build_model()
model.compile(optimizer='adam',
              loss=dice_loss,
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

# ========== Test on One Sample ==========
test_image_path = r"C:\Users\Dhanush\Desktop\LRDE INTERNSHIP\main code\segmentation_data\image.png"
test_mask_path = r"C:\Users\Dhanush\Desktop\LRDE INTERNSHIP\main code\segmentation_data\mask\ground_truth_mask.png"

test_img = cv2.imread(test_image_path)
test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

test_img = cv2.resize(test_img, IMAGE_SIZE).astype('float32') / 255.0
test_mask = cv2.resize(test_mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

test_img_tensor = np.expand_dims(test_img, axis=0)
test_pred = model.predict(test_img_tensor)
test_pred_mask = np.argmax(test_pred, axis=-1).squeeze()

# ========== Dice Score Calculation ==========
def calculate_dice_per_class(pred_mask, true_mask, num_classes=6):
    scores = []
    for i in range(num_classes):
        pred_i = (pred_mask == i).astype(np.float32)
        true_i = (true_mask == i).astype(np.float32)
        intersection = np.sum(pred_i * true_i)
        dice = (2. * intersection + 1e-7) / (np.sum(pred_i) + np.sum(true_i) + 1e-7)
        scores.append(dice)
    return scores

dice_scores = calculate_dice_per_class(test_pred_mask, test_mask, NUM_CLASSES)
print("Test Dice Scores:", dice_scores)

# ========== Reporting ==========
print("Classification Report:")
print(classification_report(test_mask.flatten(), test_pred_mask.flatten()))

print("Confusion Matrix:")
print(confusion_matrix(test_mask.flatten(), test_pred_mask.flatten()))

