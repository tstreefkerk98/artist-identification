import numpy as np
import matplotlib.pyplot as plt
import random


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


if __name__ == '__main__':
    # Example of the different variables
    model_labels = ['model1', 'model2']
    models_values = [[random.random() for _ in range(57)] for _ in range(len(model_labels))]
    artist_labels = [f"artist {i + 1}" for i in range(57)]

    plot_artist_performance_bar_chart_vertical(models_values, model_labels, artist_labels)
    plot_artist_performance_bar_chart_horizontal(models_values, model_labels, artist_labels)
