from Common import load_data
from PIL import Image

data = load_data("data")

for i in range(0,3):
    img = Image.open(data[i][0])
    img.show()
