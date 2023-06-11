import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from tqdm import tqdm

from BaselineCNN import BaselineCNN
from util import *

Image.MAX_IMAGE_PIXELS = 933120000
cudnn.benchmark = True
plt.ion()  # interactive mode
seed = 1
torch.random.manual_seed(seed)
np.random.seed(seed)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'wikiart_dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####
# Hyperparameters
#####
learning_rate = 1e-3
num_epochs = 3
batch_size = 16
betas = (0.9, 0.999)
epsilon = 1e-8


def train_model(name, model, criterion, optimizer_init, num_epochs=25, pretrained=True):
    since = time.time()
    optimizer = optimizer_init(model.parameters() if not pretrained else model.fc.parameters(), learning_rate)
    statistics = {}

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        train_loss = sys.maxsize
        train_acc = 0
        val_loss = sys.maxsize
        val_acc = 0
        converged = False

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    if train_loss < loss:
                        if converged:
                            return model, statistics, optimizer
                        converged = True
                        optimizer = optimizer_init(model.parameters(), learning_rate / 10)
                    train_loss = loss
                    train_acc = torch.sum(preds == labels.data) / len(preds)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    val_loss = epoch_loss
                    val_acc = epoch_acc

            write_statistics(statistics, name, seed, epoch, train_loss, train_acc, val_loss, val_acc)
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, statistics, optimizer


def run(name, model, num_ftrs, num_epochs, pretrained, seed):
    timestamp = get_timestamp()
    fingerprint = f"{name}_{seed}_{timestamp}"
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_init = lambda parameters, learning_rate: optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.9999))

    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 57), nn.Softmax(1))
        model.fc.requires_grad = True

    model = model.to(device)

    model_trained, statistics, optimizer = train_model(name, model, criterion, optimizer_init,
                                                       num_epochs=num_epochs, pretrained=pretrained)

    save_model(model, optimizer, name, seed, timestamp, fingerprint)

    save_statistics(statistics, fingerprint)

    return model_trained, statistics


def main():
    seeds = range(1)
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        # resnet_imagenet = models.resnet18(weights='IMAGENET1K_V1')
        # num_ftrs_imagenet = resnet_imagenet.fc.in_features
        # run("resnet18_imagenet", resnet_imagenet, num_ftrs_imagenet, num_epochs, pretrained=True, seed=seed)
        #
        # resnet_cifar10 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        # num_ftrs_cifar10 = resnet_cifar10.fc.in_features
        # run("resnet18_cifar10", resnet_cifar10, num_ftrs_cifar10, num_epochs, pretrained=True, seed=seed)
        #
        # resnet_cifar100 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
        # num_ftrs_cifar100 = resnet_cifar100.fc.in_features
        # run("resnet18_cifar100", resnet_cifar100, num_ftrs_cifar100, num_epochs, pretrained=True, seed=seed)

        baseline = BaselineCNN(57)
        run("baseline", baseline, -1, num_epochs, pretrained=False, seed=seed)


if __name__ == "__main__":
    main()
