import numpy as np


def Normalize(batch_img):
    batch_img = batch_img / 255.
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img
