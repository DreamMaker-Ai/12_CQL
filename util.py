import numpy as np

from PIL import Image


def preprocess(frame):
    img = Image.fromarray(frame).convert("L").resize((84, 84))
    img = np.array(img, dtype=np.float32)

    return img
