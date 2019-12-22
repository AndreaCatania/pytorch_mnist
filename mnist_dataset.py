import torch
import os
import json
from PIL import Image


class SparseMnistReader():
    def __init__(self, path, yolo_data, check_integrity = False):
        self.path = path
        self.yolo_data = yolo_data

        with open(path + '/meta.json') as f:
            self.meta = json.load(f)

        if check_integrity:
            # Check integrity
            count = 0
            while True:
                try:
                    with open(path + '/' + str(count) + '.png') as _:
                        pass
                    with open(path + '/' + str(count) + '.json') as _:
                        pass
                    count += 1
                except FileNotFoundError:
                    break
    
            if count != self.size():
                exit("The directory is not integral, you can generate it using the `generate_dataset.py` script.")


    """ Returns the length of the dataset """
    def size(self):
        return self.meta['size']


    """ Returns the image width """
    def image_width(self):
        return self.meta['image_width']


    """ Returns the image height """
    def image_height(self):
        return self.meta['image_height']

    
    """ Returns the raw byte array """
    def get_byte_image(self, image_id):
        if image_id >= self.size():
            exit(str(image_id) + "out of index")
        
        return Image.open(self.path + '/' + str(image_id) + '.png').tobytes()
        

    """ Returns the Raw metadata array """
    def get_image_meta(self, image_id):
        if image_id >= self.size():
            exit(str(image_id) + "out of index")
        
        with open(self.path + '/' + str(image_id) + '.json', 'r') as f:
            data = json.load(f)
        
        return data
    

    """ Get an array of floats that rapresent the image """
    def get_image(self, image_id):
        image = self.get_byte_image(image_id)
        return [float(p) / 255.0 for p in image]
    

    """ Returns a quad_index where this center fall off """
    def determine_quad(self, x, y):
        # Takes the coords of the quads, offsets these by the half of
        # the width and height so that the quads centers are at the middle
        # Then just take the quad with the shortest distance from that point.
        quads_coords = self.yolo_data[0]
        quad_half_w = self.yolo_data[1] * 0.5
        quad_half_h = self.yolo_data[2] * 0.5

        dist = 1e4
        quad_index = -1
        for index, q in enumerate(quads_coords):
            r_x = x - (q[0] + quad_half_w)
            r_y = y - (q[1] + quad_half_h)
            r_dist = r_x * r_x + r_y * r_y
            if r_dist < dist:
                dist = r_dist
                quad_index = index

        return quad_index


    """ Get offset from quad """
    def get_quad_offset(self, quad_index, x, y):
        quads_coords = self.yolo_data[0]

        quad_w = self.yolo_data[1]
        quad_h = self.yolo_data[2]

        r_x = x - quads_coords[quad_index][0]
        r_y = y - quads_coords[quad_index][1]

        return r_x / quad_w, r_y / quad_h


    """ Get targets in Yolo format:
        yolo_quds * [Confidence Any, Center X, Center Y, Box W, Box H, Confidence 0, Confidence 1, Confidence 2, Confidence 3, Confidence 4, Confidence 5, Confidence 6, Confidence 7, Confidence 8, Confidence 9]
    """
    def get_targets(self, image_id):
        meta = self.get_image_meta(image_id)

        targets = [None] * len(self.yolo_data[0])
        for v, _ in zip(meta, range(len(self.yolo_data[0]))):
            number = int(v[0])
            classes = [1.0 if i==number else 0.0 for i in range(10)]

            # Offset to the quad position
            center_x = float(v[1])
            center_y = float(v[2])
            quad_index = self.determine_quad(center_x, center_y)
            offset_x, offset_y = self.get_quad_offset(quad_index, center_x, center_y)

            # Normalize to image size
            box_w = float(v[3]) / self.image_width()
            box_h = float(v[4]) / self.image_height()

            has_object_confidence = 1.0
            
            # Add these to the target tensor
            targets[quad_index] = [has_object_confidence, offset_x, offset_y, box_w, box_h]
            targets[quad_index].extend(classes)
        
        # Set 0.0 to the void quads
        for i in range(len(targets)):
            if targets[i] == None:
                targets[i] = [0.0] * 15

        return [v for t in targets for v in t]
        

class SparseMnistDataset(torch.utils.data.Dataset):
    def __init__(self, path, yolo_quads):
        super(SparseMnistDataset, self).__init__()

        self.reader = SparseMnistReader(
            path, yolo_quads
        )
    

    def __len__(self):
        return self.reader.size()
    

    def __getitem__(self, image_id):
        image = self.reader.get_image(image_id)
        targets = self.reader.get_targets(image_id)
        return torch.tensor(image)\
            .view((
                1,
                self.reader.image_width(),
                self.reader.image_height()
            )),\
            torch.tensor(targets)


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

    
    """ Returns an array of bytes that contains a Mnist image """
    def get_byte_image(self, image_id):
        image_offset_from = image_id * 28 * 28
        image_offset_to = image_offset_from + 28 * 28

        return self.images[image_offset_from:image_offset_to]


    """ Returns thr actual number rapresented in the image pointed by `image_id` """
    def get_number(self, image_id):
        return self.labels[image_id]


    """ Returns an array of floats that contains a Mnist image """
    def get_image(self, image_id):
        image = self.get_byte_image(image_id)
        return [float(p) / 255.0 for p in image]


    """
        Returns an array of 10 floats, with all values set to 0 except for the
        value at position rapresented in the image pointed by `image_id` which is 1
    """
    def get_target(self, image_id):
        number = self.get_number(image_id)
        return [1.0 if v == number else 0.0 for v in range(10)]


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
