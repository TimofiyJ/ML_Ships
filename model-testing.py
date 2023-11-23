# Dependencies
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image

# Define the Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    return (2.0 * intersection + smooth) / (union + smooth)

# Initialize paths
data_path = './airbus-ship-detection/' # directory of the data
test_dir = data_path+'test_v2/' # Directory where the test images are currently located

# Load the model 
custom_objects = {'dice_coefficient': dice_coefficient}
loaded_model = tf.keras.models.load_model('./save/model', custom_objects=custom_objects)

# Test the model
test_dir_os = os.listdir(test_dir)
image_path = test_dir+test_dir_os[6]
print(image_path)
image = Image.open(image_path)
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)

predicted_mask = loaded_model.predict(image_array)
threshold = 0.5
predicted_mask = (predicted_mask > threshold).astype(int)
image_array = np.squeeze(image_array, axis=0)

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title('Original Image')

# Plot the predicted mask
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask[0][:, :, 0])  # Assuming masks are grayscale
plt.title('Predicted Mask')

plt.show()