import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import Common
import math, random
import cv2


def random_rotate(img, steering_angle):
    angle = np.random.uniform(-7.5, 7.5)
    return img.rotate(angle, resample=Image.BICUBIC, expand = 0), steering_angle - angle / 25.0


def random_translate(img, steering_angle):
    delta_x = np.random.uniform(-40, 40)
    delta_y = np.random.uniform(-10, 10)
    img = img.transform(img.size, Image.AFFINE, (1.0, 0.0, delta_x, 0.0, 1.0, 0.0))
    angle = math.atan2(delta_x , 95.0) * 180 / math.pi / 25
    return img,steering_angle + angle


def random_shadow(img, steering_angle):
    (w,h) = img.size
    if random.random() > 0.5:
        p1 = [0, random.uniform(0,h/2)]
    else:
        p1 = [random.uniform(0,w/2), 0]

    if random.random() > 0.5:
        p2 = [0, random.uniform(h/2,h)]
    else:
        p2 = [random.uniform(0,w/2), h]

    if random.random() > 0.5:
        p3 = [w, random.uniform(h/2,h)]
    else:
        p3 = [random.uniform(w/2,w), h]

    if random.random() > 0.5:
        p4 = [w, random.uniform(0,h/2)]
    else:
        p4 = [random.uniform(w/2,w), 0]

    intensity = random.uniform(0.0,0.6)

    img_data = cv2.cvtColor(np.asarray(img).copy(),cv2.COLOR_RGB2HLS)
    shadow_img = img_data.copy().astype(np.uint8)
    cv2.fillPoly(shadow_img, np.array([[p1,p2,p3,p4]], np.int32), (0,0,0))
    img_data[:,:,1] = (1.0 - intensity) * img_data[:,:,1] + intensity * shadow_img[:,:,1]
    img_data = cv2.cvtColor(img_data, cv2.COLOR_HLS2RGB)
    return Image.fromarray(img_data), steering_angle


def random_mirror(img, steering_angle):
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        steering_angle *= -1

    return img, steering_angle


def DataGenerator(data, batch_size=64, augment_data=True):
    num_data = len(data)
    idx = 0
    recovery_angle_a = 3.5
    recovery_angle_b = 7.5
    side_camera_angle = 1.5

    while True:
        X = []
        y = []
        for i in range(0,batch_size):
            while True:
                file_name, telemetry = data[idx]
                angle = telemetry.steering_angle
                throttle = telemetry.throttle
                idx = (idx + 1) % num_data
                if True or angle >= 0.01 or random.random() < 0.125:
                    break

            img = Image.open(file_name)
            dir_name,_,file_name = file_name.split('/')

            if dir_name.startswith("recovery_left"):
                angle += recovery_angle_b / 25.0
            elif dir_name.startswith("recovery_right"):
                angle -= recovery_angle_b / 25.0

            if dir_name.startswith("lane_left"):
                angle += recovery_angle_a / 25.0
            elif dir_name.startswith("lane_right"):
                angle -= recovery_angle_a / 25.0

            if "right" in file_name:
                angle -= side_camera_angle / 25.0
            elif "left" in file_name:
                angle += side_camera_angle / 25.0

            if (augment_data):
                img, angle = random_rotate(img, angle)
                img, angle = random_translate(img, angle)
                img, angle = random_shadow(img, angle)

            img, angle = random_mirror(img, angle)
            img_data = Common.preprocess_image(img)
            X.append(img_data)
            y.append(angle)

        yield np.array(X), np.array(y)


if __name__ == '__main__':
    num_cols = 8 * 5
    num_rows = 10 * 3
    w = 32
    h = 32

    dirs = "center_01"#,recovery_left_01,recovery_right_01,lane_left_01,lane_right_01"
    #dirs = "center_01"
    data = []
    for d in dirs.split(','):
        data += Common.load_data(d)

    #random.shuffle(data)

    overview_img = Image.new("RGBA",(w * num_cols, h * num_rows), (0,0,0))
    gen = DataGenerator(data, batch_size=num_rows * num_cols)
    X,y = gen.__next__()
    draw = ImageDraw.Draw(overview_img)

    for j in range(num_rows):
        for i in range(num_cols):
            idx = j*num_cols + i
            xi = i*w
            yi = j*h
            img = Image.fromarray(cv2.cvtColor((X[idx] * 255.0).astype('uint8'), cv2.COLOR_HLS2RGB))
            angle = y[idx]
            overview_img.paste(img, (xi,yi))
            draw.text((xi,yi), "%.2f" % angle, fill = (255,0,0))

    overview_img.show()
