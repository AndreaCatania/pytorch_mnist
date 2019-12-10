import torch

class MnistTrainDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_path):
        super(MnistTrainDataset, self).__init__()

        with open(mnist_path + "/train-images-idx3-ubyte", "rb") as f:
            images_raw_data = f.read()

        with open(mnist_path + "/train-labels-idx1-ubyte", "rb") as f:
            labels_raw_data = f.read()

        self.items_count = int.from_bytes(images_raw_data[4:8], "big")
        self.height = int.from_bytes(images_raw_data[8:12], "big")
        self.width = int.from_bytes(images_raw_data[12:16], "big")

        self.images = images_raw_data[16:]
        self.labels = labels_raw_data[8:]
    
    def __len__(self):
        return self.items_count

    
    def __getitem__(self, image_id):
        image_offset_from = image_id * 28 * 28
        image_offset_to = image_offset_from + 28 * 28

        letter = self.labels[image_id]

        image = [float(p) / 255.0 for p in self.images[image_offset_from:image_offset_to]]

        target = [1.0 if v == letter else 0.0 for v in range(10)]

        return torch.tensor(image).view((1, 28, 28)), torch.tensor(target)
        

"""
Test dataset.
It returns the Test image and the actual letter written in the image.
"""
class MnistTestDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_path):
        super(MnistTestDataset, self).__init__()

        with open(mnist_path + "/t10k-images-idx3-ubyte", "rb") as f:
            images_raw_data = f.read()

        with open(mnist_path + "/t10k-labels-idx1-ubyte", "rb") as f:
            labels_raw_data = f.read()

        self.items_count = int.from_bytes(images_raw_data[4:8], "big")
        self.height = int.from_bytes(images_raw_data[8:12], "big")
        self.width = int.from_bytes(images_raw_data[12:16], "big")

        self.images = images_raw_data[16:]
        self.labels = labels_raw_data[8:]
    

    def __len__(self):
        return self.items_count

    
    def __getitem__(self, image_id):
        image_offset_from = image_id * 28 * 28
        image_offset_to = image_offset_from + 28 * 28

        letter = self.labels[image_id]

        image = [float(p) / 255.0 for p in self.images[image_offset_from:image_offset_to]]

        return torch.tensor(image).view((1, 28, 28)), letter
