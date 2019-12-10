
import torch

import letter_vision_model as lvm
import mnist_dataset as md

if not torch.cuda.is_available():
    exit("Cuda is not available.")

# Test dataset
dataset = md.MnistTestDataset("./resources/mnist")

# Load model
model = lvm.LetterVisionModel()
model.load_state_dict(torch.load("./resources/models/mnist_model_weights.pt"))
model.eval()
model = model.cuda()

# Test the model:
for image_id in range(100):
    data = dataset[image_id]

    out = model(data[0].view(1, 1, 28, 28).cuda()).cpu()
    accuracy = out[0][data[1]].item()

    print(
        "Is value: "
        + str(data[1])
        + " corectly predicted? "
        + ("Yes" if accuracy > 0.5 else "No")
        + " with accuracy: "
        + str(accuracy))

