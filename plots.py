import numpy as np
import matplotlib.pyplot as plt
import random

from main import load_statistics


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
    bar_height = 0.3
    for i, model_values in enumerate(models_values):
        y = np.arange(num_artists) + i * bar_height
        plt.barh(y, model_values, height=bar_height, edgecolor='grey', label=model_labels[i], align='center')

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
            del seed['test_loss']
            del seed['test_acc']
            train_losses.append([seed[epoch]['Training loss'] for epoch in seed])
            train_accuracies.append([seed[epoch]['Training accuracy'] for epoch in seed])
            val_losses.append([seed[epoch]['Validation loss'] for epoch in seed])
            val_accuracies.append([seed[epoch]['Validation accuracy'] for epoch in seed])

        mean_train_losses, std_train_losses = mean_and_std(train_losses)
        mean_train_accuracies, std_train_accuracies = mean_and_std(train_accuracies)
        mean_val_losses, sdt_val_losses = mean_and_std(val_losses)
        mean_val_accuracies, std_val_accuracies = mean_and_std(val_accuracies)

        x_axis = range(len(mean_train_losses))
        if len(statistics) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.errorbar(x_axis, mean_train_losses, yerr=std_train_losses, label='Training loss', ecolor=blue)
        ax.errorbar(x_axis, mean_train_accuracies, yerr=std_train_losses, label='Training accuracy', ecolor=orange)
        ax.errorbar(x_axis, mean_val_losses, yerr=sdt_val_losses, label='Validation loss', ecolor=green)
        ax.errorbar(x_axis, mean_val_accuracies, yerr=std_val_accuracies, label='Validation accuracy', ecolor=red)
        ax.set(xlabel='Epoch', ylabel='Value')
        ax.set_title(model_key)
        ax.legend()
    plt.show()


if __name__ == '__main__':
    plot_with_std = False
    opacity = 0.5 if plot_with_std else 0.0
    blue = [c / 256 for c in [75, 146, 194]] + [opacity]
    orange = [c / 256 for c in [255, 142, 42]] + [opacity]
    green = [c / 256 for c in [85, 178, 85]] + [opacity]
    red = [c / 256 for c in [222, 81, 82]] + [opacity]

    # Example for `plot_run_values`
    plot_training_statistics(['resnet18_imagenet1k_1_1_20230609-100202', 'resnet18_imagenet1k_2_2_20230610-000017', 'resnet18_imagenet1k_3_3_20230610-010624', 'resnet18_imagenet1k_4_4_20230610-021353', 'resnet18_imagenet1k_5_5_20230610-031505'])

    # # Example of the different variables
    # model_labels = ['model1', 'model2']
    # models_values = [[random.random() for _ in range(57)] for _ in range(len(model_labels))]
    # artist_labels = [f"artist {i + 1}" for i in range(57)]
    #
    # plot_artist_performance_bar_chart_vertical(models_values, model_labels, artist_labels)
    # plot_artist_performance_bar_chart_horizontal(models_values, model_labels, artist_labels)
