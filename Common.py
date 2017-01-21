import pickle
from Telemetry import Telemetry
from PIL import Image, ImageOps, ImageFilter
import numpy as np


def load_data(dir):
    with open(dir + "/labels.pickle", "rb") as f:
        return pickle.load(f)


def preprocess_image(img):
    img = img.crop((0,70,320,134))
    img = img.resize((32,32), resample=Image.BICUBIC)
    return np.array(img).astype('float32') / 255.0
