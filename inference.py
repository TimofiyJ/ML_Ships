# Import Dependencies
import tensorflow as tf
import sys 
import os
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

# Define the Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    return (2.0 * intersection + smooth) / (union + smooth)


# Get the arguments
directory_location = os.listdir(str(sys.argv[1]))

# Load the model
custom_objects = {'dice_coefficient': dice_coefficient}
loaded_model = tf.keras.models.load_model('./model-1', custom_objects=custom_objects)

save_folder_location = './results'

# Save the results
for i in directory_location:
    image = Image.open(os.path.join(directory_location, i))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    prediction = loaded_model.predict(image_array)
    threshold = 0.5
    predicted_mask = (prediction > threshold).astype(int)
    predicted_mask.save(os.path.join(save_folder_location, i))
