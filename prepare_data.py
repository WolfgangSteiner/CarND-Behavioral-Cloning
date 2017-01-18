import pickle
import pandas
import argparse
import Utils
from Telemetry import Telemetry

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()

data = pandas.read_csv(args.dir + "/" + "driving_log.csv")

new_data = []
i = 0

def path_name(img_file):
    return args.dir + '/' + img_file.strip()

for r in data.iterrows():
    Utils.progress_bar(i+1, len(data))
    i+=1
    center_img = path_name(r[1].center)
    left_img = path_name(r[1].left)
    right_img = path_name(r[1].right)
    steering = r[1].steering
    throttle = r[1].throttle
    brake = r[1].brake
    speed = r[1].speed

    telemetry = Telemetry(steering, throttle, brake, speed)
    new_data.append((center_img, telemetry))
    new_data.append((left_img, telemetry))
    new_data.append((right_img, telemetry))


with open(args.dir + "/labels.pickle", "wb") as f:
    pickle.dump(new_data,f)
