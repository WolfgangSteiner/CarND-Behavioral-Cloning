import pickle
from Telemetry import Telemetry

def load_data(dir):
    with open(dir + "/labels.pickle") as f:
        return pickle.load(f)
