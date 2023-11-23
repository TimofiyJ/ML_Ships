# Dependencies
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

# Main functions
def decode_run_length(encoded_coordinates):
    pixel_numbers = []
    pixel_number = np.array(encoded_coordinates[::2], dtype=int) 
    pixel_amount = np.array(encoded_coordinates[1::2], dtype=int)
    
    for i in range(len(pixel_amount)):
        for j in range(pixel_number[i], pixel_number[i] + pixel_amount[i]):
            pixel_numbers.append(j)

    pixel_numbers = np.array(pixel_numbers)
    
    return pixel_numbers


def create_mask(pixels_list, image_size=(768, 768)):
    # Create a blank image with black background
    mask = Image.new('L', image_size, color=0)
    mask_array = np.array(mask)
    pixel_numbers = decode_run_length(pixels_list)
    np.set_printoptions(threshold=np.inf)
    x, y = np.unravel_index(pixel_numbers.astype(int)-1, image_size)
    # Set pixels to white in the mask
    mask_array[y, x] = 255
    return mask_array

def get_image(index,dataframe,pixels_dictionary):
    '''
    get np.array of image
    param index: index of the dataset row
    param dataframe: dataframe that has ImageId and EncodedPixels columns
    returns: image(np.array)
    '''
    # ImageId - column with path to the image in the training directory
    # EncodedPixels - column with run-length encoded pixels of the ship
    image_path = dataframe.iloc[index]['ImageId']
    pixels_list = pixels_dictionary[image_path].split() 

    image = Image.open(train_dir + image_path)

    image_array = np.array(image)

    return image_array

def get_mask(index,dataframe,pixels_dictionary):
    '''
    get np.array image's mask
    param index: index of the dataset row
    param dataframe: dataframe that has ImageId and EncodedPixels columns
    param pixels_dictionary: dictionary with key - ImageId and value - all EncodedPixels of ships that this Image has
    returns: mask(np.array)
    '''
    # ImageId - column with path to the image in the training directory
    # EncodedPixels - column with run-length encoded pixels of the ship
    image_path = dataframe.iloc[index]['ImageId']
    pixels_list = pixels_dictionary[image_path].split() 

    if (not pd.isna(dataframe.iloc[index]['EncodedPixels'])):
        
        pixels_list = [int(x) for x in pixels_list]

        mask_array = create_mask(pixels_list, (768, 768))

        mask_array = mask_array.astype(int)

        mask_array = np.expand_dims(mask_array, axis=-1)

        return mask_array
    else:
        
        mask_array = create_mask([0], (768, 768))
        mask_array = np.expand_dims(mask_array, axis=-1)

        return mask_array



# Define the U-Net architecture
def unet_model(input_size=(768, 768, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Middle
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv3])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv2])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv1])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    return (2.0 * intersection + smooth) / (union + smooth)

# Define data generator
class CustomDataGenerator(Sequence):
    def __init__(self, dataframe, pixels_dictionary, batch_size=32, img_size=(768, 768), augment=False):
        self.dataframe = dataframe
        self.pixels_dictionary = pixels_dictionary
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_image_indices = range(start_idx, min(end_idx, len(self.dataframe)))

        # Load and preprocess images and masks
        batch_images = [self.get_image(index) for index in batch_image_indices]
        batch_masks = [self.get_mask(index) for index in batch_image_indices]

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        # Shuffle the dataframe after each epoch
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def get_image(self, index):
        image = get_image(index, self.dataframe, self.pixels_dictionary)  # Replace with your actual get_image function
        image = image / 255.0  # Normalize to [0, 1]

        # Apply data augmentation if needed
        if self.augment:
            # Add your augmentation logic here
            pass

        return image

    def get_mask(self, index):
        mask = get_mask(index, self.dataframe, self.pixels_dictionary)  # Replace with your actual get_mask function
        mask = mask / 255.0  # Normalize to [0, 1]

        # Apply data augmentation if needed
        if self.augment:
            # Add your augmentation logic here
            pass

        return mask
    

# Initialize paths
data_path = './airbus-ship-detection/' # directory of the data
train_csv ='./airbus-ship-detection/train_ship_segmentations_v2.csv'  # Path to the CSV train file 
test_dir = data_path+'test_v2/' # Directory where the test images are currently located
train_dir = data_path+'train_v2/' # Directory where the train images are currently located

# Get the data
df = pd.read_csv(train_csv)     

# Get unique images
df_unique_images = df.copy()
df_unique_images['NumShips'] = df_unique_images.groupby('ImageId')['ImageId'].transform('count')
df_unique_images.loc[df_unique_images['EncodedPixels'].isna(), 'NumShips'] = 0
df_unique_images = df_unique_images.drop_duplicates(subset=['ImageId'])

# Undersample the data
no_ships_samples = 100
one_ship_samples = 1710
more_than_one_ship_samples = 945

undersampled_no_ships = df_unique_images[df_unique_images['NumShips'] == 0].sample(no_ships_samples)
undersampled_one_ship = df_unique_images[df_unique_images['NumShips'] == 1].sample(one_ship_samples)
undersampled_more_than_one_ship = df_unique_images[df_unique_images['NumShips'] > 1].sample(more_than_one_ship_samples)

undersampled_df = pd.concat([undersampled_no_ships, undersampled_one_ship, undersampled_more_than_one_ship])

undersampled_df = undersampled_df.sample(frac=1).reset_index(drop=True)

# All ships in one image
pixels_dict = {} # key - path to image value - pixels of all the ships in this image

for _, row in df.iterrows():
    image_id = row['ImageId']
    encoded_pixels = str(row['EncodedPixels'])
    
    if image_id not in pixels_dict:
        pixels_dict[image_id] = encoded_pixels
    else:
        pixels_dict[image_id] += ' ' + encoded_pixels

# Test - validation split
X_train, X_val = train_test_split(undersampled_df, 
    test_size=0.25, random_state= 8) 

# Hyperparameters
image_size = (768,768)
num_epochs = 3
batch_size = 4
steps_per_epoch = 5
validation_steps = 5

train_generator = CustomDataGenerator(X_train, pixels_dictionary=pixels_dict, batch_size=batch_size, augment=False)
val_generator = CustomDataGenerator(X_val, pixels_dictionary=pixels_dict, batch_size=batch_size, augment=False)


model = unet_model(input_size=(768, 768, 3))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coefficient])
model.summary()

# Train the model
history = model.fit(train_generator,batch_size = batch_size, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val_generator, validation_steps=validation_steps)

# Save the model
model.save('./model-1')