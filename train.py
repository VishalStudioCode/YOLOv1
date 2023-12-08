# Purpose: create training loop for YOLOv1 model
import torch
import torch.optim as optim
import torch.utils.data as data
from VOC import VocDataset
from model import YoloV1
from yolov1_loss import YoloLoss
import pathlib

# save weights to this Path
PATH = pathlib.Path(".") / "architecture" / "state.pt";
PATH.parent.mkdir(parents=True, exist_ok=True);
PATH.touch(exist_ok=True);

if(torch.cuda.is_available()):
    device = ("cuda");
elif(torch.backends.mps.is_available()):
    device = ("mps");
else:
    device = ("cpu");

# Need to import YOLO model and YOLO loss function from other files
# Karthik Selvaraj 11/6/2022 7:41 PM 

# Training function that predicts using model and training data, calculates loss and does backpropagation
# Karthik Selvaraj 11/6/2022 7:42 PM 
def train(model, optimizer, train_loader, loss_func):
    size = len(train_loader.dataset);
    # Puts model in training mode and creates variable to keep track of total loss in batches
    model.train()
    total_loss = 0

    for batch, (x_train, y_train) in enumerate(train_loader):
        # Predicts based on x_train and calculates loss
        x_train, y_train = x_train.to(device), y_train.to(device);
        y_pred = model(x_train)
        loss = loss_func(y_pred, y_train)

        # Zeros optimizer and does backpropagtion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adds to total loss of the epoch
        total_loss += loss.item()
        if(batch % 10 == 0):
            loss, current = loss.item(), (batch + 1) * len(x_train);
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss

# Testing function used to test how model functions using testing data (not written yet)
# Karthik Selvaraj 11/6/2022 7:45 PM 
def test():
    pass

def main():
    print(f"Using {device} device");

    # Intializes model from architecture file and uses SGD optimizer for backpropagation
    # Karthik Selvaraj 11/6/2022 7:47 PM

    model = YoloV1().to(device);
    # load our model if possible
    # if our file is not empty
    if(PATH.stat().st_size != 0):
        model.load_state_dict(torch.load(str(PATH.absolute())));
        print("Successfully loaded model's previous state");
    optimizer = optim.SGD(model.parameters(), lr=0.001);
    loss = YoloLoss();
    # Sets total number of epochs and creates train/test data loaders
    # Karthik Selvaraj 11/6/2022 7:48 PM 
    epochs = 30;

    train_dataset = VocDataset();

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = data.DataLoader()

    print(f"Number of training images: {len(train_dataset)}");

    # Training loop that calls training function for specified number of epochs 
    # Karthik Selvaraj 11/6/2022 7:49 PM 
    for epoch in range(epochs):
        cur_loss = train(model, optimizer, train_loader, loss) / len(train_loader)
        print(f'Epoch {epoch}: {cur_loss}')
    # save the model
    torch.save(model.state_dict(), str(PATH.absolute()));


if __name__ == "__main__":
    main()