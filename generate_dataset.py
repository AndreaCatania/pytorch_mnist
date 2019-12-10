from PIL import Image

import mnist_dataset as md

dataset = md.MnistDataReader(
    "./resources/mnist/train-images-idx3-ubyte",
    "./resources/mnist/train-labels-idx1-ubyte"
)

image = dataset.get_byte_image(0)
number = dataset.get_target(0)

img = Image.frombytes(data=image, mode='L', size=(28, 28))
img.show()
print(number)

