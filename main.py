import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from tqdm import tqdm

TESTING = True

# Path to datasets
dataset_dir = "dataset"

NUM_CLASSES = 57

# Hyperparameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 10
batch_size = 32
betas = (0.9, 0.999)
epsilon = 1e-8


def train():
    # Set the device to be used (cuda if available, else cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for data preprocessing
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = ImageFolder(dataset_dir, transform=data_transforms)

    print('length: ', len(dataset))

    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset)) if not TESTING else 57 * 4
    val_size = int(0.1 * len(dataset)) if not TESTING else 57
    test_size = int(0.1 * len(dataset)) if not TESTING else 57
    void_size = len(dataset) - train_size - val_size - test_size
    train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size, test_size, void_size])

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load a pre-trained ResNet-50 model
    model = resnet50(pretrained=True)
    num_classes = len(dataset.classes)

    # num_classes should equal 57
    assert num_classes == NUM_CLASSES

    # Modify the last fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon)

    # Training loop
    print('Starting training loop...\n')
    for _ in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        train_loss /= len(train_loader)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = test_correct / test_total
        test_loss /= len(test_loader)

        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    train()
