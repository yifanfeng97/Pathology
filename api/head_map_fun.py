from PIL import Image
import numpy as np


def get_heat_map_from_prob(prob):
    prob = prob[:, :, np.newaxis]
    heat_map_np = np.concatenate((255*prob, np.zeros(prob.shape), 255*(1-prob)), axis=2).astype(np.uint8)
    return Image.fromarray(heat_map_np)