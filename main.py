import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
from datetime import datetime
import PIL
import os

SAVE_MODEL = True
PIL.Image.MAX_IMAGE_PIXELS = 933120000
SAVE_STATISTICS = True
TESTING = False
NUM_CLASSES = 57

#####
# Path to datasets
#####
dataset_dir = 'dataset'
models_dir = 'models'
statistics_dir = 'statistics'

#####
# Hyperparameters
#####
learning_rate = 1e-3
num_epochs = 100
batch_size = 128
betas = (0.9, 0.999)
epsilon = 1e-8
num_workers = 3 * os.cpu_count() // 4
# Number of epochs until convergence is assumed
early_stop_limit = 5
early_stop_epsilon = 1e-2
seeds = [1]


def main(seed=1):
    # Set the device to be used (cuda if available, else cpu).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for data preprocessing.
    data_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = ImageFolder(dataset_dir, transform=data_transforms)

    # Split the dataset into training, validation, and test sets.
    if TESTING:
        train_size, val_size, test_size = NUM_CLASSES * 4, NUM_CLASSES, NUM_CLASSES
    else:
        train_size, val_size, test_size = int(0.8 * len(dataset)), int(0.1 * len(dataset)), int(0.1 * len(dataset))
    void_size = len(dataset) - train_size - val_size - test_size

    # If we are not testing the void_size should be zero.
    if not TESTING:
        assert void_size == 0

    train_dataset, val_dataset, test_dataset, _ = \
        torch.utils.data.random_split(dataset, [train_size, val_size, test_size, void_size])

    # Create data loaders for training, validation, and test sets.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load a pre-trained ResNet-18 model.
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_classes = len(dataset.classes)

    # num_classes should equal 57.
    assert num_classes == NUM_CLASSES

    # Modify the last fully connected layer to match the number of classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the device.
    model = model.to(device)

    # Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon)
    last_train_acc = None
    epochs_without_improvement = 0
    converged = False

    # Object to store train and validation losses and accuracies in.
    statistics = {}

    # TODO: Set seed and model_name dynamically
    model_name = 'resnet18-imagenet'

    timestamp = get_timestamp()
    # The fingerprint of the run, for example: 'resnet18-imagenet_123_20230601-125524'
    fingerprint = f"{model_name}_{seed}_{timestamp}"

    # Training loop
    print('Starting training loop...\n')
    reporter = tqdm(range(num_epochs))
    for epoch in reporter:
        message = ''
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

        if last_train_acc is not None and last_train_acc - train_loss < early_stop_epsilon:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        message += f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} | "

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        val_loss /= len(val_loader)

        message += f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        reporter.set_description(message)

        if epochs_without_improvement == early_stop_limit:
            # optimizer.lr /= 10
            # Converged before -> stop completely
            if converged:
                print("Training finished.")
                break
            print("Last layer converged.")
            # Start training the entire model
            for param in model.parameters():
                param.requires_grad = True
            converged = True

        last_train_acc = train_accuracy

        # Write train and validation losses and accuracies to `statistics` object.
        if SAVE_STATISTICS:
            write_statistics(statistics, model_name, seed, epoch, train_loss, train_accuracy, val_loss, val_accuracy)

    # If `statistics` is not empty, it means we stored some stats, and we should save them.
    if statistics:
        save_statistics(statistics, fingerprint)

    # Save model.
    if SAVE_MODEL:
        save_model(model, optimizer, model_name, seed, timestamp, fingerprint)


def save_model(model, optimizer, model_name, seed, timestamp, fingerprint):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_name': model_name,
        'seed': seed,
        'timestamp': timestamp,
    }, f"{models_dir}/{fingerprint}.pt")


def load_model(model, optimizer, fingerprint, evaluate=True):
    state = torch.load(f"{models_dir}/{fingerprint}.pt")
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    model_name = state['model_name']
    seed = state['seed']

    # Depending on purpose of model loading we call either `eval()` or `train()`.
    if evaluate:
        model.eval()
    else:
        model.train()

    return model_name, seed


def write_statistics(statistics, model_name, seed, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    if model_name not in statistics:
        statistics[model_name] = {}
    if seed not in statistics[model_name]:
        statistics[model_name][seed] = {}
    statistics[model_name][seed][epoch] = {
        'Training loss': train_loss,
        'Training accuracy': train_accuracy,
        'Validation loss': val_loss,
        'Validation accuracy': val_accuracy,
    }


def save_statistics(statistics, fingerprint):
    with open(f"{statistics_dir}/{fingerprint}.pkl", 'wb') as outp:
        pickle.dump(statistics, outp, pickle.HIGHEST_PROTOCOL)


def load_statistics(fingerprint):
    with open(f"{statistics_dir}/{fingerprint}.pkl", 'rb') as inp:
        return pickle.load(inp)


def get_timestamp():
    now = datetime.now()
    fill = lambda x: str(x).zfill(2)
    return f"{now.year}{fill(now.month)}{fill(now.day)}-{fill(now.hour)}{fill(now.minute)}{fill(now.second)}"


if __name__ == '__main__':
    print("cuda" if torch.cuda.is_available() else "cpu")
    for seed in seeds:
        print(f"\n\nRUNNING SEED {seed}\n\n")
        main(seed)
