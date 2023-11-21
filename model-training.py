from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
TF_ENABLE_ONEDNN_OPTS=0
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageFile

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
    image_path = dataframe.loc[index, 'ImageId']
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
    image_path = dataframe.loc[index, 'ImageId']
    pixels_list = pixels_dictionary[image_path].split() 

    if (not pd.isna(dataframe.loc[index, 'EncodedPixels'])):
        
        pixels_list = [int(x) for x in pixels_list]

        mask_array = create_mask(pixels_list, (768, 768))
                
        mask_array = np.expand_dims(mask_array, axis=-1)

        return mask_array
    else:
        
        mask_array = create_mask([], (768, 768))
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


data_path = './airbus-ship-detection/' # directory of the data
train_csv ='./airbus-ship-detection/train_ship_segmentations_v2.csv'  # Path to the CSV train file 
test_dir = data_path+'test_v2/' # Directory where the test images are currently located
train_dir = data_path+'train_v2/' # Directory where the train images are currently located

df = pd.read_csv(train_csv)     

df_unique_images = df.copy()
df_unique_images['NumShips'] = df_unique_images.groupby('ImageId')['ImageId'].transform('count')
df_unique_images.loc[df_unique_images['EncodedPixels'].isna(), 'NumShips'] = 0
df_unique_images = df_unique_images.drop_duplicates(subset=['ImageId'])


no_ships_samples = 26112
one_ship_samples = 27104
more_than_one_ship_samples = 15452

undersampled_no_ships = df_unique_images[df_unique_images['NumShips'] == 0].sample(no_ships_samples)
undersampled_one_ship = df_unique_images[df_unique_images['NumShips'] == 1].sample(one_ship_samples)
undersampled_more_than_one_ship = df_unique_images[df_unique_images['NumShips'] > 1].sample(more_than_one_ship_samples)

undersampled_df = pd.concat([undersampled_no_ships, undersampled_one_ship, undersampled_more_than_one_ship])

undersampled_df = undersampled_df.sample(frac=1).reset_index(drop=True)

pixels_dict = {}

for _, row in df.iterrows():
    image_id = row['ImageId']
    encoded_pixels = str(row['EncodedPixels'])
    
    if image_id not in pixels_dict:
        pixels_dict[image_id] = encoded_pixels
    else:
        pixels_dict[image_id] += ' ' + encoded_pixels


class CustomDataGenerator:
    def __init__(self, dataframe, pixels_dictionary, batch_size, img_size=(768, 768)):
        self.dataframe = dataframe
        self.pixels_dictionary = pixels_dictionary
        self.batch_size = batch_size
        self.img_size = img_size

        # ImageDataGenerator for augmentation
        self.data_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            cval=0
        )

    def generate_batches(self):
        while True:
            batch_indices = np.random.choice(len(self.dataframe), size=self.batch_size, replace=False)
            batch_image_paths = self.dataframe['ImageId'].iloc[batch_indices].tolist()

            images = []
            masks = []

            for image_path in batch_image_paths:
                image_array, mask_array = self.get_data(image_path)
                images.append(image_array)
                masks.append(mask_array)

            yield np.array(images), np.array(masks)

    def get_data(self, image_path):
        index = self.dataframe[self.dataframe['ImageId'] == image_path].index[0]

        # Load and resize image
        image = Image.open(train_dir + image_path)
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape((1,) + image_array.shape)

        # Load and resize mask
        mask_array = get_mask(index, self.dataframe, self.pixels_dictionary)
        mask_array = mask_array.reshape((1,) + mask_array.shape + (1,))

        # Perform data augmentation
        seed = np.random.randint(999999)
        params = self.data_generator.get_random_transform(image_array.shape[1:], seed=seed)
        augmented_image_array = self.data_generator.apply_transform(image_array[0], params)
        augmented_mask_array = self.data_generator.apply_transform(mask_array[0, :, :, 0], params)

        return augmented_image_array, augmented_mask_array

batch_size = 32
epochs = 20
img_size = (768, 768)

data_generator = CustomDataGenerator(dataframe=undersampled_df, pixels_dictionary=pixels_dict, batch_size=batch_size, img_size=img_size)


model = unet_model(input_size=(768, 768, 3))

# # Compile the model with the Dice coefficient metric
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[dice_coefficient])
model.summary()
num_epochs = 10  # Adjust the number of epochs as needed
steps_per_epoch = len(undersampled_df)//batch_size
model.fit(data_generator.generate_batches(), steps_per_epoch=steps_per_epoch, epochs=num_epochs)
tf.keras.Model.save(model)


# # Print a summary of the model architecture



