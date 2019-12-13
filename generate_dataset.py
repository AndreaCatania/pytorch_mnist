import math
import random
import json
import argparse
import os
from PIL import Image

import mnist_dataset as md

MNIST_SIZE = 28
MNIST_HALF_SIZE = 28 / 2

class MnistAugmentPipeline():
    def __init__(self, canvas_size):
        self.dataset = md.MnistDataReader(
            "./resources/mnist/train-images-idx3-ubyte",
            "./resources/mnist/train-labels-idx1-ubyte"
        )

        self.w = canvas_size
        self.h = canvas_size
        self.min_dist = math.sqrt(pow(MNIST_SIZE, 2)*2)
        self.try_count = 20


    def overlap(self, center_a, center_b):
        delta = center_b[0] - center_a[0], center_b[1] - center_a[1]
        size = math.sqrt(pow(delta[0], 2) + pow(delta[1], 2))
        return self.min_dist >= size


    def get_centers(self):
        # Calculates the margin to be sure that the letter are necer clipped
        inner_x = MNIST_SIZE / 2
        inner_y = MNIST_SIZE / 2
        inner_w = self.w - inner_x
        inner_h = self.h - inner_y

        # Generate a non overlapping centers
        centers = []
        for _desired_centers in range(random.randrange(2, 5)):
            for _ in range(self.try_count):
                center = random.randint(inner_x, inner_w), random.randint(inner_y, inner_h)
                overlap = False
                for c in centers:
                    if self.overlap(c, center):
                        overlap = True
                        break
                if not overlap:
                    centers.append(center)
                    break
        
        return centers


    def generate(self, amount, path):

        for generated_img_id in range(amount):
            centers = self.get_centers()
            numbers = []

            canvas = bytearray(self.w * self.h)

            for c in centers:
                image_id = random.randint(0, self.dataset.len() - 1)

                number_image = self.dataset.get_byte_image(image_id)
                numbers.append(self.dataset.get_number(image_id))

                cnv_ofs_x = c[0] - MNIST_SIZE / 2
                cnv_ofs_y = c[1] - MNIST_SIZE / 2
                for x in range(MNIST_SIZE):
                    for y in range(MNIST_SIZE):
                        canvas[int(self.h * (cnv_ofs_y + y) + (cnv_ofs_x + x))] = number_image[MNIST_SIZE * y + x]

            name = str(generated_img_id)

            # Store Image
            img = Image.frombytes(
                data=bytes(canvas),
                mode='L',
                size=(self.w, self.h))
            img.save(path + '/' + name + '.png')

            # Store data info
            data = []
            for c, n in zip(centers, numbers):
                data.append([n, c[0], c[1], MNIST_HALF_SIZE, MNIST_HALF_SIZE])

            with open(path + '/' + name + '.json', 'w') as f:
                json.dump(data, f)

        meta = {
            'image_meta_format': '[number, center_x, center_y, extent_x, extent_y]',
            'size': amount,
            'image_channels': 1,
            'image_width': 84,
            'image_height': 84,
        }
        with open(path + '/meta.json', 'w') as f:
            json.dump(meta, f)


parser = argparse.ArgumentParser(description="Generates a dataset of images 84x84 hand written digits, and a relative bounding box.")

parser.add_argument(
    "--count",
    default=0,
    type=int,
    help="The amount of images to generates.")

parser.add_argument(
    "--path",
    default="./resources/sparse_mnist",
    type=str,
    help="The amount of images to generates.")

arg = parser.parse_args()

if len(os.listdir(arg.path) ) != 0:
    exit("The destination directory is not empty")

aug = MnistAugmentPipeline(MNIST_SIZE * 3)
aug.generate(arg.count, arg.path)