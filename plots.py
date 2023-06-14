import os

import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
from torchvision import datasets, models
from torchvision.transforms import transforms

from BaselineCNN import BaselineCNN
from util import load_statistics, load_model
from collections import defaultdict
from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000


def plot_artist_performance_bar_chart_vertical(models_values, model_labels, artist_labels):
    """
    Creates a vertical bar chart plot
    :param models_values: list of D by 57, where D is the amount of models
    :param model_labels: list of D model labels
    :param artist_labels: list of 57 artist labels
    :return: void
    """
    num_artists = len(models_values[0])
    bar_width = 0.25
    for i, model_values in enumerate(models_values):
        x = np.arange(num_artists) + i * bar_width
        plt.bar(x, model_values, width=bar_width, edgecolor='grey', label=model_labels[i])

    plt.xlabel('Artists', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xticks([dx + bar_width for dx in range(num_artists)], artist_labels, rotation=90)
    plt.legend()
    plt.ylim([0, 1])
    plt.show()


def plot_artist_performance_bar_chart_horizontal(models_values, model_labels, artist_labels):
    """
    Creates a horizontal bar chart plot
    :param models_values: list of D by 57, where D is the amount of models
    :param model_labels: list of D model labels
    :param artist_labels: list of 57 artist labels
    :return: void
    """
    num_artists = len(models_values[0])
    bar_height = 0.2
    plt.figure(figsize=(20, 30))
    for i, model_values in enumerate(models_values):
        y = np.arange(num_artists) + i * bar_height
        plt.barh(y, model_values, height=bar_height, edgecolor='grey', label=model_labels[i], align='edge')

    plt.ylabel('Artists', fontweight='bold')
    plt.xlabel('Accuracy', fontweight='bold')
    plt.yticks([dy + bar_height for dy in range(num_artists)], artist_labels)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlim([0, 1])
    plt.show()


def plot_training_statistics(paths):
    """
    Plots all the models averaged over the seeds it finds in the `paths` variable.
    :param paths: The paths to statistics files, for example:
    `['resnet18_imagenet1k_1_1_20230606-081824', 'resnet18_imagenet1k_1_2_20230606-081824']`
    :return: void
    """
    statistics = {}
    for path in paths:
        new_statistics = load_statistics(path)
        for model_key in new_statistics:
            if model_key not in statistics:
                statistics[model_key] = new_statistics[model_key]
            else:
                for seed_key in new_statistics[model_key]:
                    if seed_key not in statistics[model_key]:
                        statistics[model_key][seed_key] = new_statistics[model_key][seed_key]
                    else:
                        for epoch_key in new_statistics[model_key][seed_key]:
                            if epoch_key not in statistics[model_key][seed_key]:
                                statistics[model_key][seed_key][epoch_key] = new_statistics[model_key][seed_key][
                                    epoch_key]
                            else:
                                print('This has data for the same epoch, something is wrong: ' + path)

    fig, axes = plt.subplots(nrows=1, ncols=len(statistics))

    def mean_and_std(xs):
        return np.mean(xs, axis=0), np.std(xs, axis=0)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for i, model_key in enumerate(statistics):
        model = statistics[model_key]

        for seed_key in model:
            seed = model[seed_key]
            # del seed['test_loss']
            # del seed['test_acc']
            train_losses.append([seed[epoch]['Training loss'] for epoch in seed])
            train_accuracies.append([seed[epoch]['Training accuracy'].cpu() for epoch in seed])
            val_losses.append([seed[epoch]['Validation loss'] for epoch in seed])
            val_accuracies.append([seed[epoch]['Validation accuracy'].cpu() for epoch in seed])

        mean_train_losses, std_train_losses = mean_and_std(train_losses)
        mean_train_accuracies, std_train_accuracies = mean_and_std(train_accuracies)
        mean_val_losses, sdt_val_losses = mean_and_std(val_losses)
        mean_val_accuracies, std_val_accuracies = mean_and_std(val_accuracies)

        x_axis = range(len(mean_train_losses))
        plt.errorbar(x_axis, mean_train_losses, yerr=std_train_losses, label='Training loss', ecolor=blue)
        plt.errorbar(x_axis, mean_val_losses, yerr=sdt_val_losses, label='Validation loss', ecolor=green)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title("Losses " + model_key)
        plt.legend()
        plt.show()
        plt.errorbar(x_axis, mean_train_accuracies, yerr=std_train_losses, label='Training accuracy', ecolor=orange)
        plt.errorbar(x_axis, mean_val_accuracies, yerr=std_val_accuracies, label='Validation accuracy', ecolor=red)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        plt.title("Accuracies " + model_key + " | Last accuracy: " + str(round(mean_val_accuracies[-1], 2)))
        plt.legend()
        plt.show()


def test_models(models):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'wikiart_dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over data.
    acc = defaultdict(lambda: defaultdict(float))
    for name, model in models.items():
        model.to(device)
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(len(preds)):
                predicted_artist = class_names[preds[i]]
                label_artist = class_names[labels[i]]
                if predicted_artist == label_artist:
                    # Each artist has 30 images in the test set
                    acc[name][label_artist] += 1 / 30

    return acc, class_names


if __name__ == '__main__':
    plot_with_std = False
    opacity = 0.5 if plot_with_std else 0.0
    blue = [c / 256 for c in [75, 146, 194]] + [opacity]
    orange = [c / 256 for c in [255, 142, 42]] + [opacity]
    green = [c / 256 for c in [85, 178, 85]] + [opacity]
    red = [c / 256 for c in [222, 81, 82]] + [opacity]

    fingerprints = {
        "baseline": 'baseline_8_20230614-120141',
        "resnet18": 'resnet18_imagenet_8_20230614-101748',
        "resnet20-10": 'resnet20_cifar10_8_20230614-104930',
        "resnet20-100": 'resnet20_cifar100_8_20230614-112509'
    }

    # # Plot loss graphs
    # # Baseline
    # plot_training_statistics([fingerprints["baseline"]])
    # # ImageNet1k
    # plot_training_statistics([fingerprints["resnet18"]])
    # # Cifar10
    # plot_training_statistics([fingerprints["resnet20-10"]])
    # # Cifar100
    # plot_training_statistics([fingerprints["resnet20-100"]])

    # Example of the different variables
    baseline = BaselineCNN()
    resnet18 = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs_imagenet = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs_imagenet, 57)
    resnet20_10 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    num_ftrs_cifar10 = resnet20_10.fc.in_features
    resnet20_10.fc = nn.Linear(num_ftrs_cifar10, 57)
    resnet20_100 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
    num_ftrs_cifar100 = resnet20_100.fc.in_features
    resnet20_100.fc = nn.Linear(num_ftrs_cifar100, 57)
    models = {
        "baseline": load_model(baseline, fingerprints["baseline"])[0],
        "resnet18-imagenet": load_model(resnet18, fingerprints["resnet18"])[0],
        "resnet20-cifar10": load_model(resnet20_10, fingerprints["resnet20-10"])[0],
        "resnet20-cifar100": load_model(resnet20_100, fingerprints["resnet20-100"])[0]
    }
    models_values, artist_labels = test_models(models)
    # artist_labels = ['Fernand Leger', 'Erte', 'Sam Francis']
    # artist_labels = ['Alfred Sisley', 'Albert Bierstadt', 'Isaac Levitan', 'Ivan Shishkin', 'Camille Corot', 'Ivan Aivazovsky', 'Claude Monet']
    artist_labels = ['Francisco Goya', 'Rembrandt', 'Pierre-Auguste Renoir', 'Ilya Repin', 'William Merritt Chase', 'Amedeo Modigliani', 'Zdislav Beksinski', 'Raphael Kirchner']
    model_names = list(models.keys())
    reconstructed_values = [[models_values[name][artist] for artist in artist_labels] for name in model_names]

    plot_artist_performance_bar_chart_horizontal(reconstructed_values, model_names, artist_labels)
