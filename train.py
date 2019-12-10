import torch
import torchvision
import torch.utils.tensorboard as tb

import letter_vision_model as lvm
import mnist_dataset as md

if not torch.cuda.is_available():
    exit("Cuda is not available!")

summary_writer = tb.SummaryWriter()

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
    lr = 0.3,
    momentum=0.9,
    weight_decay=0.0001
)

graph_not_logged = True

# Log graph
def onetime_log_graph(epoch, images):
    global model 
    global summary_writer
    global graph_not_logged

    if graph_not_logged:
        summary_writer.add_image(
            'batch_sample',
            torchvision.utils.make_grid(images),
            epoch
        )

        summary_writer.add_graph(
            model,
            images
        )

        graph_not_logged = False


def log_epoch(epoch, loss, accuracy):
    global summary_writer

    summary_writer.add_scalar('epoch_loss', loss, epoch)
    summary_writer.add_scalar('epoch_accuracy', accuracy, epoch)


def log_model(epoch):
    global model 
    global summary_writer

    for name, param in model.named_parameters():
        summary_writer.add_histogram(
            name,
            param,
            epoch)

        summary_writer.add_histogram(
            name + ".grad",
            param.grad,
            epoch)


def step_train_epoch(epoch):

    preload_count = 10
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

        onetime_log_graph(epoch, images)

        train_count += 1

        optmizer.zero_grad()
        predictions = model(images)
        loss = mean(predictions, targets)
        loss.backward()
        optmizer.step()

        total_loss += loss.item()
        total_correctly_pred += md.correctly_predicted_count(predictions, targets)

    return total_loss / train_count, total_correctly_pred / len(dataset)


for epoch in range(5):
    loss, accuracy = step_train_epoch(epoch)
    log_epoch(epoch, loss, accuracy)
    log_model(epoch)


torch.save(model.state_dict(), "./resources/models/mnist_model_weights.pt")
summary_writer.close()