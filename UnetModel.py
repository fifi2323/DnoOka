import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    return Model(inputs, output)


def preprocess_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale
    img = cv2.resize(img, img_size)  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (H, W, 1)
    return img


# Load Data
train_images, train_masks = [], []
val_images, val_masks = [], []

image_files = sorted(glob.glob("healthy/*.jpg"))  # Image path
mask_files = sorted(glob.glob("healthy_vains/*.tif"))  # Expert mask path

for img_path, mask_path in zip(image_files, mask_files):
    img = preprocess_image(img_path)
    mask = preprocess_image(mask_path)

    if len(train_images) < 13:
        train_images.append(img)
        train_masks.append(mask)
    else:
        val_images.append(img)
        val_masks.append(mask)

# Convert to NumPy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)
val_images = np.array(val_images)
val_masks = np.array(val_masks)

# Reshape masks to match model output (H, W, 1)
train_masks = train_masks.reshape(-1, 256, 256, 1)
val_masks = val_masks.reshape(-1, 256, 256, 1)

# Model Training
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

history = model.fit(train_images, train_masks,
                    validation_data=(val_images, val_masks),
                    epochs=50, batch_size=2, callbacks=callbacks)  # Reduced batch size due to small dataset
model.save("unet_retinal_vessel.h5")