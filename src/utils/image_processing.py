import PIL
from PIL import Image
import os

def get_image(config, filename: str) -> PIL.Image.Image:
    path = os.path.join(
        config.dataset.root,
        config.dataset.img_folder,
        filename
    )
    img = Image.open(path)
    return img
