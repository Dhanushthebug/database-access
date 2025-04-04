import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Input, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths for training data
train_images_path = "path/to/train/images"
train_masks_path = "path/to/train/masks"

def load_data(image_dir, mask_dir, img_size=(512, 512)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    images = []
    masks = []
    
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(image_dir, img_file))
        img = cv2.resize(img, img_size) / 255.0
        
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load dataset
train_images, train_masks = load_data(train_images_path, train_masks_path)

# Expand dimensions for model compatibility
train_masks = np.expand_dims(train_masks, axis=-1)

# Build FCN-8s model
def build_fcn_model():
    inputs = Input(shape=(512, 512, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(6, (16, 16), strides=(8, 8), padding='same', activation='softmax')(x)
    return Model(inputs, x)

fcn_model = build_fcn_model()
fcn_model.compile(optimizer=Adam(learning_rate=0.0002), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train FCN-8s model
fcn_model.fit(train_images, train_masks, epochs=10, batch_size=8, validation_split=0.2)

# Save trained FCN model
fcn_model.save("fcn_model.h5")

# DC-GAN Training

def build_generator():
    inputs = Input(shape=(100,))
    x = Dense(8 * 8 * 256, activation='relu')(inputs)
    x = Reshape((8, 8, 256))(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(1, (16, 16), strides=(32, 32), padding='same', activation='sigmoid')(x)
    return Model(inputs, x)

def build_discriminator():
    inputs = Input(shape=(512, 512, 1))
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs, x)

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

# GAN Model
z = Input(shape=(100,))
generated_mask = generator(z)
validity = discriminator(generated_mask)
gan = Model(z, validity)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Prepare GAN training data
train_masks_gan = train_masks / 255.0  # Normalize masks
train_masks_gan = np.expand_dims(train_masks_gan, axis=-1)

# Training loop
batch_size = 16
epochs = 100
half_batch = batch_size // 2

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, train_masks_gan.shape[0], half_batch)
    real_masks = train_masks_gan[idx]
    
    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_masks = generator.predict(noise)
    
    real_labels = np.ones((half_batch, 1))
    fake_labels = np.zeros((half_batch, 1))
    
    d_loss_real = discriminator.train_on_batch(real_masks, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_masks, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Save models
generator.save("dcgan_generator.h5")
discriminator.save("dcgan_discriminator.h5")