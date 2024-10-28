import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image, target_size=(224,224)):
    
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    
    return np.expand_dims(image_array ,axis=0)

def create_data_generator(batch_size=32):
    
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        
    )
    