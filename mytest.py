
import torch
import numpy as np
import time
# Display the images using matplotlib
import matplotlib.pyplot as plt

def blackish_percent(bands, threshold=(100, 100, 100)):
    r, g, b = bands
    t = threshold
    mask = (r < t[0]) & (g < t[1]) & (b < t[2])
    percentage = mask.avg() / 255.0

    return percentage


# Assuming `image` is a PyTorch tensor of shape [C, H, W] where C is the number of channels (3 for RGB)
def split_rgb_torch(image):
    # Ensure the image tensor is on the correct device (CPU or GPU)
    # image should be a FloatTensor with values in [0, 1] or a ByteTensor with values in [0, 255]
    r = image[:,0, :, :]
    g = image[:,1, :, :]
    b = image[:,2, :, :]
    return (r, g, b)


# Define color thresholds
PENS_RGB = {
    "red": [
        (150, 80, 90),
        (110, 20, 30),
        (185, 65, 105),
        (195, 85, 125),
        (220, 115, 145),
        (125, 40, 70),
        (200, 120, 150),
        (100, 50, 65),
        (85, 25, 45),
    ],
    "green": [
        (150, 160, 140),
        (70, 110, 110),
        (45, 115, 100),
        (30, 75, 60),
        (195, 220, 210),
        (225, 230, 225),
        (170, 210, 200),
        (20, 30, 20),
        (50, 60, 40),
        (30, 50, 35),
        (65, 70, 60),
        (100, 110, 105),
        (165, 180, 180),
        (140, 140, 150),
        (185, 195, 195),
    ],
    "blue": [
        (60, 120, 190),
        (120, 170, 200),
        (120, 170, 200),
        (175, 210, 230),
        (145, 210, 210),
        (37, 95, 160),
        (30, 65, 130),
        (130, 155, 180),
        (40, 35, 85),
        (30, 20, 65),
        (90, 90, 140),
        (60, 60, 120),
        (110, 110, 175),
    ],

    "black": [
        (100, 100, 100)]
}



def pen_percent_torch(bands, pen_color):
    r, g, b = bands
    thresholds = PENS_RGB[pen_color]
    mask = torch.zeros_like(r, dtype=torch.bool, device='cuda')

    if pen_color == "red":
        t = thresholds[0]
        mask = (r > t[0]) & (g < t[1]) & (b < t[2])
        for t in thresholds[1:]:
            mask = mask | ((r > t[0]) & (g < t[1]) & (b < t[2]))

    elif pen_color == "green":
        t = thresholds[0]
        mask = (r < t[0]) & (g > t[1]) & (b < t[2])
        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g > t[1]) & (b > t[2])
    elif pen_color == "blue":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b > t[2])
        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g < t[1]) & (b > t[2])
    elif pen_color == "black":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b < t[2])

    else:
        raise Exception(f"Error: pen_color='{pen_color}' not supported")

    # Compute the mean across all images in the batch
    #percentage = mask.float().mean(dim=[1, 2]) / 255.0 # Average over each image's spatial dimensions
    percentage = torch.sum(mask).item()/mask.numel()

    #percentage = mask.numel()/255
    #pen_percent = (pen_pixels / total_pixels) * 100


    return percentage







import pyvips
def pen_percent(bands, pen_color):
    r, g, b = bands
    thresholds = PENS_RGB[pen_color]

    if pen_color == "red":
        t = thresholds[0]
        mask = (r > t[0]) & (g < t[1]) & (b < t[2])

        for t in thresholds[1:]:
            mask = mask | ((r > t[0]) & (g < t[1]) & (b < t[2]))


    elif pen_color == "green":
        t = thresholds[0]
        mask = (r < t[0]) & (g > t[1]) & (b < t[2])
        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g > t[1]) & (b > t[2])

    elif pen_color == "blue":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b > t[2])
        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g < t[1]) & (b > t[2])

    else:
        raise Exception(f"Error: pen_color='{pen_color}' not supported")

    percentage = mask.avg() / 255.0

    return percentage


import os
import numpy as np
import time
from multiprocessing import Pool
import pyvips

# Code setting up functions and constants

def display_image(tensor, title):
    plt.imshow(tensor[0].permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'

    import torch

    # Define the dimensions of the image
    height, width = 100, 100  # You can set these to whatever dimensions you need

    # Create tensors for each color
    black_image = torch.zeros(1, 3, height, width, dtype=torch.uint8)
    blue_image = torch.zeros(1, 3, height, width, dtype=torch.uint8)
    blue_image[:, 2, :, :] = 255  # Set the blue channel to 255
    red_image = torch.zeros(1, 3, height, width, dtype=torch.uint8)
    red_image[:, 0, :, :] = 255  # Set the red channel to 255
    green_image = torch.zeros(1, 3, height, width, dtype=torch.uint8)
    green_image[:, 1, :, :] = 255  # Set the green channel to 255


    bands = split_rgb_torch(black_image)
    colors = ["red", "green", "blue", "black"]
    results = {color: pen_percent_torch(bands, color) for color in colors}
    red_per = results["red"]
    green_per = results["green"]
    blue_per = results["blue"]
    black_per = results["black"]

    # ink_removed = np.asarray(visuals["fake_B"][0].permute(1, 2, 0).cpu().numpy() * 255, np.uint8)

    real_ink = np.asarray(black_image[0].permute(1, 2, 0).numpy(), np.uint8)
    # compute percentage of red, blue, gree, and black
    real_ink_vips = pyvips.Image.new_from_array(real_ink)
    bands = real_ink_vips.bandsplit()

    colors = ["red", "green", "blue"]
    perc = []
    for color in colors:
        percentage = pen_percent(bands, color)
        perc.append(percentage)
    percentage = blackish_percent(bands)
    perc.append(percentage)

    print()

