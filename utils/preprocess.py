import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array
