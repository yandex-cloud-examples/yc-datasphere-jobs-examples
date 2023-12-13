import argparse
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
from model import GarmentClassifier

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(prog='test')
parser.add_argument('-r', '--result', required=True, help='Output file')

if __name__ == '__main__':
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=4, shuffle=False)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(epoch_index, tb_writer=None):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                if tb_writer:
                    tb_x = epoch_index * len(training_loader) + i + 1
                    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5
    best_vloss = 1000000

    latest_model = None

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}'.format(epoch_number)
            torch.save(model.state_dict(), model_path)
            latest_model = model_path

        epoch_number += 1

    if latest_model:
        print(f"Move {latest_model} to {args.result}")
        shutil.move(latest_model, args.result)
