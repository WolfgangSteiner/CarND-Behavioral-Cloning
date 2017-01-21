import pickle
import pandas
import argparse
import Utils
from Telemetry import Telemetry

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()

data = open(args.dir + "/" + "driving_log.csv","r").readlines()

new_data = []
i = 0

def path_name(img_path):
    return '/'.join(img_path.split('/')[-3:])

for l in data:
    Utils.progress_bar(i+1, len(data))
    i+=1
    center_img,left_img,right_img,steering,throttle,brake,speed = l.split(", ")
    center_img = path_name(center_img)
    left_img = path_name(left_img)
    right_img = path_name(right_img)
    telemetry = Telemetry(float(steering), float(throttle), float(brake), float(speed))
    new_data.append((center_img, telemetry))
    new_data.append((left_img, telemetry))
    new_data.append((right_img, telemetry))


with open(args.dir + "/labels.pickle", "wb") as f:
    pickle.dump(new_data,f)
