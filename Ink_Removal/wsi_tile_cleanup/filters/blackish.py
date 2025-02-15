import pyvips as vips
import numpy as np


def blackish_percent(bands, threshold=(100, 100, 100)):
    r, g, b = bands
    t = threshold
    mask = (r < t[0]) & (g < t[1]) & (b < t[2])
    percentage = mask.avg() / 255.0

    return percentage
