import torch
import torchvision
import torch.utils.tensorboard as tb
from PIL import Image

import yolo_model as ym
import mnist_dataset as md

print(torch.__version__)

if not torch.cuda.is_available():
    exit("Cuda is not available!")
else:
    print("Cuda is available")

summary_writer = tb.SummaryWriter()

model = ym.YoloModel().train().cuda()

dataset = md.SparseMnistDataset(
    "./resources/sparse_mnist",
    model.yolo_data(84, 84),
)
data_loader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = 1, 
    shuffle = True,
    num_workers = 1, # Prefetch the images
    pin_memory = True,
)


mean = torch.nn.MSELoss()
graph_not_logged = True

def step_train_epoch(epoch):

    preload_count = 5
    preloads = []

    loader_iter = iter(data_loader)

    total_loss = 0.0
    train_count = 0

    total_correctly_pred = 0

    while True:
        for _, batch in zip(range(preload_count - len(preloads)), loader_iter):
            preloads.append(
                (
                    batch[0].cuda(non_blocking = True),
                    batch[1].cuda(non_blocking = True)
                )
            )

        if len(preloads) == 0:
            break
        
        (images, targets) = preloads.pop(0)

        train_count += 1

        print(images.size())
        out = model(images)
        print(out)
        print(targets)
        exit("Exit <-")

        #optmizer.zero_grad()
        #predictions = model(images)
        #loss = mean(predictions, targets)
        #loss.backward()
        #optmizer.step()

        #total_loss += loss.item()
        #total_correctly_pred += md.correctly_predicted_count(predictions, targets)

    return total_loss / train_count, total_correctly_pred / len(dataset)

step_train_epoch(0)