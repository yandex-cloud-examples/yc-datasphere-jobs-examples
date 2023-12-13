import argparse
import os
import shutil

import keras_cv
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('prompt', type=str)


if __name__ == '__main__':
    args = parser.parse_args()

    generated_dir = "generated_images"
    os.makedirs(generated_dir, exist_ok=True)

    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    images = model.text_to_image(args.prompt, batch_size=3)
    for i in range(len(images)):
        im = Image.fromarray(images[i])
        im.save(f"./{generated_dir}/image_{i}.jpeg")

    shutil.make_archive("generated_images", "zip", ".", generated_dir)
