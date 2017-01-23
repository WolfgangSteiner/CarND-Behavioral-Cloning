import pickle
from Telemetry import Telemetry
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

def load_data(dir):
    with open(dir + "/labels.pickle", "rb") as f:
        return pickle.load(f)


def preprocess_image(img):
    img = img.crop((0,70,320,134))
    img = img.resize((32,32), resample=Image.BICUBIC)
    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2HLS)
    return img.astype('float32') / 255.0
