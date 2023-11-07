# Purpose: create training loop for YOLOv1 model
import torch
import torch.optim as optim
import torch.utils.data as data

# Need to import YOLO model and YOLO loss function from other files
# Karthik Selvaraj 11/6/2022 7:41 PM 

# Training function that predicts using model and training data, calculates loss and does backpropagation
# Karthik Selvaraj 11/6/2022 7:42 PM 
def train(model, optimizer, train_loader, loss_func):
    # Puts model in training mode and creates variable to keep track of total loss in batches
    model.train()
    total_loss = 0

    for _, (x_train, y_train) in enumerate(train_loader):
        # Predicts based on x_train and calculates loss
        y_pred = model(x_train)
        loss = loss_func(y_pred, y_train)

        # Zeros optimizer and does backpropagtion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adds to total loss of the epoch
        total_loss += loss.item()

    return total_loss

# Testing function used to test how model functions using testing data (not written yet)
# Karthik Selvaraj 11/6/2022 7:45 PM 
def test():
    pass

def main():
    # Intializes model from architecture file and uses SGD optimizer for backpropagation
    # Karthik Selvaraj 11/6/2022 7:47 PM

    # model = YOLOv1()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Sets total number of epochs and creates train/test data loaders
    # Karthik Selvaraj 11/6/2022 7:48 PM 
    epochs = 100

    train_loader = data.DataLoader()
    test_loader = data.DataLoader()

    # Training loop that calls training function for specified number of epochs 
    # Karthik Selvaraj 11/6/2022 7:49 PM 
    for epoch in range(epochs):
        # cur_loss = train(model, optimizer, train_loader, YOLO_loss) / len(train_loader)
        # print(f'Epoch {epoch}: {cur_loss}')
        pass

if __name__ == "__main__":
    main()