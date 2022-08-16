
from PIL import Image
import numpy as np

def load_image(path):
    return np.array(Image.open(path))

print(load_image("/opt/notebooks/dataset/DIV2K_train_HR/0001.png").shape)
print(load_image("/opt/notebooks/dataset/DIV2K_train_LR_bicubic/X4/0001x4.png").shape)

print(load_image("/opt/notebooks/dataset/Projeto/HR/0.png").shape)
print(load_image("/opt/notebooks/dataset/Projeto/LR/0.png").shape)
print(384/96)
print(1404/351)
