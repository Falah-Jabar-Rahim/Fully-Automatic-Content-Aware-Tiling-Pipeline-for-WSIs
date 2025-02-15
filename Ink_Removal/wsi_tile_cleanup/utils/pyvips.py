import pyvips as vips


def read_image(image_path, discard_alpha=True, **kwargs):
    vips_image = vips.Image.new_from_file(f"{image_path}", **kwargs)
    
    if discard_alpha:
        # n=3 discards alpha channel, keeping only r,g,b
        vips_image = vips_image.extract_band(0, n=3)
    
    return vips_image


def split_rgb(vips_image):
    bands = vips_image.bandsplit()
    return bands

# Assuming `image` is a PyTorch tensor of shape [C, H, W] where C is the number of channels (3 for RGB)
def split_rgb_torch(image):
    # Ensure the image tensor is on the correct device (CPU or GPU)
    # image should be a FloatTensor with values in [0, 1] or a ByteTensor with values in [0, 255]
    r = image[:,0, :, :]
    g = image[:,1, :, :]
    b = image[:,2, :, :]
    return (r, g, b)