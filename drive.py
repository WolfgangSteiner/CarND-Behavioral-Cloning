import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image, ImageOps, ImageDraw
from flask import Flask, render_template
from io import BytesIO
import Common

import sys
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
target_speed = 25


def draw_mark(draw, x_rel, height=1.0, color=(255,255,255), w=320, h=160, width=1):
    abs_height = int(h / 16 * height)
    x_abs = w/2 + int(x_rel * (w/2-20))
    draw.line((x_abs, h - 40 - abs_height/2, x_abs, h - 40 + abs_height/2),fill=color,width=width)


def draw_grid(draw):
    for x in (-1.0, 0.0, 1.0):
        draw_mark(draw,x,height=2.0)

    for x in (-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8):
        draw_mark(draw,x)

    for x in (-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9):
        draw_mark(draw,x,height=0.5)


def load_model():
    global model
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)


@sio.on('telemetry')
def telemetry(sid, data):
    # Receive telemetry:
    speed = float(data["speed"])
    throttle = float(data["throttle"])
    imgString = data["image"]

    # Get and preprocess camera image:
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    X = Common.preprocess_image(image).reshape((1,32,32,3))

    # Compute the new steering angle
    steering_angle = float(model.predict(X, batch_size=1))

    # Basic speed control:
    global target_speed
    delta_throttle = min(0.05, (target_speed - speed)/ 100.0)
    throttle = min(0.6, throttle + delta_throttle)

    # Send control to simulator:
    send_control(steering_angle, throttle)

    # Prepare display of telemetry data:
    display_image = image.resize((image.width*1,image.height*1), resample=Image.LANCZOS)

    # Print telemetry data:
    draw = ImageDraw.Draw(display_image)
    draw.text((2,0), "ANG %+.2f\nTHR %.2f\nSPD %.2f\nTGT %.2f" % (steering_angle * 25.0, throttle, speed, target_speed))

    # Visualize steering angle:
    draw_grid(draw)
    draw_mark(draw, steering_angle, height=2.0, color=(0,255,0), width=3)

    # Draw the cropped/rescaled input image for the model:
    (w,h) = display_image.size
    input_img = Image.fromarray((X[0] * 255.0).astype('uint8'))
    draw.rectangle((w - input_img.width - 1, 0, w, input_img.height), fill=(255,255,255))
    display_image.paste(input_img, (320 - input_img.width, 0))

    # Display telemetry data:
    display_image_array = cv2.cvtColor(np.asarray(display_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('Telemetry',display_image_array)

    key = cv2.waitKey(1)
    if key == 43:
        target_speed = min(30, target_speed + 5)
    elif key == 45:
        target_speed -= max(0, target_speed - 5)
    elif key == 65288:
        target_speed = 0
    elif key == ord('r'):
        print("Reloading model...")
        load_model()


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    if sys.argv[1] == "--train":
        is_training_mode = True
    else:
        parser = argparse.ArgumentParser(description='Remote Driving')
        parser.add_argument('model', type=str,
        help='Path to model definition json. Model weights should be on the same path.')
        args = parser.parse_args()
        load_model()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
