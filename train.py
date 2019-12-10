import torch

import letter_vision_model as lvm
import mnist_dataset as md

if not torch.cuda.is_available():
    exit("Cuda is not available!")

dataset = md.MnistTrainDataset("./resources/mnist")
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 24,
    pin_memory = True,
    shuffle=True
)

model = lvm.LetterVisionModel().train().cuda()
mean = torch.nn.MSELoss()
optmizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01,
)

def step_tran_epoch():
    preload_count = 10
    preloads = []

    loader_iter = iter(data_loader)
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

        optmizer.zero_grad()
        predictions = model(images)
        loss = mean(predictions, targets)
        loss.backward()
        optmizer.step()

    print(loss)

for _ in range(20):
    step_tran_epoch()

torch.save(model.state_dict(), "./resources/models/mnist_model_weights.pt")