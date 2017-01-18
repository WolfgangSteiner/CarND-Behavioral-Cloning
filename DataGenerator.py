import numpy as np
from PIL import Image, ImageOps, ImageDraw
import Common
import math, random


def random_rotate(img, steering_angle):
    angle = np.random.normal(0.0, 5.0)
    return img.rotate(angle, resample=Image.BICUBIC, expand = 0), steering_angle + angle / 25.0


def random_translate(img, steering_angle):
    delta_x = np.random.normal(0, 10)
    img = img.transform(img.size, Image.AFFINE, (1.0, 0.0, delta_x, 0.0, 1.0, 0.0))
    angle = math.atan2(delta_x , 95.0) * 180 / math.pi / 25
    return img,steering_angle - angle


def random_mirror(img, steering_angle):
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        steering_angle *= -1

    return img, steering_angle


def DataGenerator(data, batch_size=128, augment_data=True):
    num_data = len(data)
    idx = 0

    while True:
        X = []
        y = []
        for i in range(0,batch_size):
            file_name, telemetry = data[idx]
            img = Image.open(file_name)
            angle = telemetry.steering_angle

            if file_name.startswith("right"):
                angle -= 2.5 / 25
            elif file_name.startswith("left"):
                angle += 2.5 / 25

            if (augment_data):
                img, angle = random_rotate(img, angle)
                img, angle = random_translate(img, angle)
                img, angle = random_mirror(img, angle)

            img = ImageOps.equalize(img)
            img_data = np.array(img).astype('float32') / 255.0
            X.append(img_data)
            y.append(angle)
            idx = (idx + 1) % num_data

        yield np.array(X), np.array(y)


if __name__ == '__main__':
    num_cols = 8
    num_rows = 10
    w = 320
    h = 160
    overview_img = Image.new("RGBA",(w * num_cols, h * num_rows), (0,0,0))
    gen = DataGenerator(num_rows * num_cols)
    X,y = gen.next()
    draw = ImageDraw.Draw(overview_img)

    for j in range(num_rows):
        for i in range(num_cols):
            idx = j*num_cols + i
            xi = i*w
            yi = j*h
            print X[idx].shape
            print X[idx]
            img = Image.fromarray((X[idx] * 255.0).astype('uint8'))
            angle = y[idx]
            overview_img.paste(img, (xi,yi))
            draw.text((xi,yi), "%.2f" % angle, fill = (255,0,0))

    overview_img.show()
