import numpy as np
import torch
from matplotlib import pyplot as plt


def estimate_easyCNN(model, image_left, image_right, metadata):

    model.eval()
    prediction_output = model.forward(image_left, image_right, metadata)

    return prediction_output

def get_coordinate_distances(coordinates):

    dist_min = 20.0
    dist_max = 0
    distances = []

    for node1 in np.arange(start=0, stop=len(coordinates)):
        for node2 in np.arange(start=node1+1, stop=len(coordinates)):
            dist_norm = np.linalg.norm([coordinates[node1, :]-coordinates[node2, :]])
            if dist_norm == 0:
                break
            distances.append(dist_norm)
            if dist_norm > dist_max:
                dist_max = dist_norm
            elif dist_norm < dist_min:
                dist_min = dist_norm

    dist_mean = np.mean(distances)

    return dist_min, dist_max, dist_mean

def angular_error(prediction, label, num_classes):
    error = np.abs(np.angle(np.exp(1j * 2 * np.pi * (prediction - label) / num_classes))) * num_classes / (2 * np.pi)
    return error

def plot_error(df, num_classes):

    targets = range(num_classes)
    errors = [[None]] * num_classes
    predictions = [[None]] * num_classes
    for target in targets:
        predictions[target] = df.loc[df['Target'] == target]['Prediction']
        tmp = df.loc[df['Target'] == target]['Prediction']
        tmp_err = []
        for err in tmp:
            tmp_err.append(angular_error(err, target, num_classes))
        errors[target] = tmp_err

    fig, ax = plt.subplots(2, 1)
    ax[0].plot([1, 72], [0, 71], 'k--')
    ax[0].boxplot(predictions)
    ax[1].boxplot(errors)
    ax[0].set_title('Predictions')
    ax[1].set_title('Errors')
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.show()

def plot_distribution(labels, num_classes):

    target = range(num_classes)
    freq = np.zeros(num_classes)

    for label in labels:
         freq[label] += 1

    plt.bar(target, freq)
    plt.title('Target distribution')
    plt.show()

def angular_error_multisource(prediction, label, num_classes):

    # print(prediction)
    # print(label)

    error = 0
    for pred, lab in zip(prediction, label):
        error = error + np.abs(np.angle(np.exp(1j * 2 * np.pi * float(pred - lab) / num_classes))) * num_classes / (2 * np.pi)
    return error

def calculate_accuracy(df, num_classes):

    criterion = 1

    accuracy_model = 0
    accuracy_srpphat = 0
    accuracy_music = 0

    for idx in range(len(df)):

        tmp = df['Prediction'][idx]
        tmp_srpphat = df['Prediction_SRPPHAT'][idx]
        tmp_music = df['Prediction_MUSIC'][idx]
        target = df['Target'][idx]

        if angular_error(target, tmp, num_classes) <= criterion:
            accuracy_model += 1
        if angular_error(target, tmp_srpphat, num_classes) <= criterion:
            accuracy_srpphat += 1
        if angular_error(target, tmp_music, num_classes) <= criterion:
            accuracy_music += 1

    return accuracy_model/len(df)*100, accuracy_srpphat/len(df)*100, accuracy_music/len(df)*100