import glob

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU

def visualize_patches(patches, num_patches=10):
    """Visualize the extracted patches in a grid."""
    fig, axes = plt.subplots(1, num_patches, figsize=(15, 5))
    for i, patch in enumerate(patches):
        try:
            axes[i].imshow(patch.squeeze(), cmap='gray')  # Remove channel dimension and display
            axes[i].axis('off')
        except:
            pass
    plt.show()

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

def preprocess_image(image_path, image_path_mask, img_size=(512, 512)):
    """Load and preprocess an image into a fixed number of patches."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale
    img_mask = cv2.imread(image_path_mask, cv2.IMREAD_GRAYSCALE)  # Load grayscale

    # Pad the image to ensure it is divisible by the patch size
    h, w = img.shape
    pad_h = (img_size[0] - h % img_size[0]) % img_size[0]
    pad_w = (img_size[1] - w % img_size[1]) % img_size[1]

    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    img_mask = cv2.copyMakeBorder(img_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    img_arr = []
    img_mask_arr = []
    patch_coords = []

    # Generate random patch coordinates
    """for _ in range(num_patches):
        i = np.random.randint(0, img.shape[0] - img_size[0])
        j = np.random.randint(0, img.shape[1] - img_size[1])
        patch_coords.append((i, j))"""

    for i in range(0, img.shape[0] - img_size[0], img_size[0]):
        for j in range(0, img.shape[1] - img_size[1], img_size[1]):
            patch_coords.append((i, j))

    # Extract patches and normalize brightness
    for i, j in patch_coords:
        # Extract and normalize image patch
        temp = img[i:i + img_size[0], j:j + img_size[1]]
        temp = temp / 255.0  # Normalize to [0, 1]
       # temp = normalize_brightness(temp)  # Normalize brightness
        temp = cv2.resize(temp, (256, 256))
        temp = np.expand_dims(temp, axis=-1)  # Add channel dimension
        img_arr.append(temp)

        # Extract and normalize mask patch
        temp_mask = img_mask[i:i + img_size[0], j:j + img_size[1]]
        temp_mask = temp_mask / 255.0  # Normalize to [0, 1]
        temp_mask = cv2.resize(temp_mask, (256, 256))
        temp_mask = np.expand_dims(temp_mask, axis=-1)  # Add channel dimension
        img_mask_arr.append(temp_mask)

    return img_arr, img_mask_arr


print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# Load Data
train_images, train_masks = [], []
val_images, val_masks = [], []

image_files = sorted(glob.glob("healthy/*.jpg"))  # Image path
mask_files = sorted(glob.glob("healthy_vains/*.tif"))  # Expert mask path

# Ensure the number of images and masks match
if len(image_files) != len(mask_files):
    raise ValueError("The number of images and masks must be the same.")


for img_path, mask_path in zip(image_files, mask_files):
    img_patches, mask_patches = preprocess_image(img_path, mask_path)
    visualize_patches(img_patches, 10)
    visualize_patches(mask_patches, 10)
    train_images.extend(img_patches)
    train_masks.extend(mask_patches)

# Convert to NumPy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)

# Split data into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=42
)

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

history = model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=30, batch_size=8, callbacks=callbacks)  # Reduced batch size due to small dataset
def visualize_predictions(model, images, masks, num_samples=5):
    """Visualize model predictions on a few samples."""
    indices = np.random.choice(len(images), num_samples, replace=False)
    for i in indices:
        # Get the image and mask
        img = images[i]
        true_mask = masks[i]

        # Predict the mask
        pred_mask = model.predict(np.expand_dims(img, axis=0))[0]
        #pred_mask = (pred_mask > 0.1).astype(np.uint8)  # Apply threshold

        # Display the results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask.squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.show()

# Visualize predictions on the validation set
visualize_predictions(model, val_images, val_masks, num_samples=5)
model.save("unet_retinal_vessel_huge_small_batch_single.h5")
def plot_training_curves(history):
    """Plot training and validation loss and accuracy."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# Plot training curves
plot_training_curves(history)
val_loss, val_accuracy = model.evaluate(val_images, val_masks)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")