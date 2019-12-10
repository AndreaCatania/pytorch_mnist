import torch

class MnistDataReader():
    def __init__(self, imgs_path, labels_path):
        with open(imgs_path, "rb") as f:
            images_raw_data = f.read()

        with open(labels_path, "rb") as f:
            labels_raw_data = f.read()

        self.items_count = int.from_bytes(images_raw_data[4:8], "big")
        self.height = int.from_bytes(images_raw_data[8:12], "big")
        self.width = int.from_bytes(images_raw_data[12:16], "big")

        self.images = images_raw_data[16:]
        self.labels = labels_raw_data[8:]


    """ Length of the dataset """
    def len(self):
        return self.items_count

    
    """ Returns an array of floats that contains a Mnist image """
    def get_image(self, image_id):
        image = self.get_byte_image(image_id)
        return [float(p) / 255.0 for p in image]


    """ Returns an array of bytes that contains a Mnist image """
    def get_byte_image(self, image_id):
        image_offset_from = image_id * 28 * 28
        image_offset_to = image_offset_from + 28 * 28

        return self.images[image_offset_from:image_offset_to]


    """
        Returns an array of 10 floats, with all values set to 0 except for the
        value at position rapresented in the image pointed by `image_id` which is 1
    """
    def get_target(self, image_id):
        number = self.get_number(image_id)
        return [1.0 if v == number else 0.0 for v in range(10)]


    """ Returns thr actual number rapresented in the image pointed by `image_id` """
    def get_number(self, image_id):
        return self.labels[image_id]


class MnistTrainDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_path):
        super(MnistTrainDataset, self).__init__()

        self.reader = MnistDataReader(
            mnist_path + "/train-images-idx3-ubyte",
            mnist_path + "/train-labels-idx1-ubyte"
        )

    
    def __len__(self):
        return self.reader.len()

    
    def __getitem__(self, image_id):
        image = self.reader.get_image(image_id)
        target = self.reader.get_target(image_id)

        return torch.tensor(image).view((1, 28, 28)), torch.tensor(target)
        

"""
Test dataset.
It returns the Test image and the actual letter written in the image.
"""
class MnistTestDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_path):
        super(MnistTestDataset, self).__init__()

        self.reader = MnistDataReader(
            mnist_path + "/t10k-images-idx3-ubyte",
            mnist_path + "/t10k-labels-idx1-ubyte"
        )


    def __len__(self):
        return self.reader.len()

    
    def __getitem__(self, image_id):
        image = self.reader.get_image(image_id)
        number = self.reader.get_number(image_id)

        return torch.tensor(image).view((1, 28, 28)), number


def correctly_predicted_count(predictions, targets):
    count = 0
    for p, t in zip(predictions, targets):
        for id in range(10):
            if p[id] >= 0.5 and t[id] >= 0.9:
                count += 1
    return count
