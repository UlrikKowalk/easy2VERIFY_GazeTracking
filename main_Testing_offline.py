import argparse
import os
import random
import subprocess
import sys
import time
import yaml
import numpy as np
import pandas as pd
import torch.nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from apply_regression import apply_regression

import Core.Evaluation as Evaluation
from GazeData import GazeData
from easyCNN_01 import easyCNN_01
from easyCNN_02 import easyCNN_02
from Core.Timer import Timer
from random import Random

with open('config_testing.yml') as config:
    configuration = yaml.safe_load(config)

save_result = configuration['save_result']
simulation_parameters = configuration['simulation_parameters']
simulation_parameters['base_dir'] = os.getcwd()



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rand_gen = torch.Generator()
    rand_gen.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return rand_gen


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    print(f'DNN: {simulation_parameters["network"]}, loading model: {simulation_parameters["model"]}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "cpu"

    trained_net = f'{simulation_parameters["base_dir"]}/{simulation_parameters["model"]}'
    print(f"Using device '{device}'.")

    dataset = GazeData(directory=simulation_parameters["dataset"], device=device)
    # dataset.set_length(1000)

    if simulation_parameters['network'] == 'easyCNN_01':
        dnn = easyCNN_01(use_metadata=simulation_parameters['use_metadata'])
    elif simulation_parameters['network'] == 'easyCNN_02':
        dnn = easyCNN_02()
    else:
        dnn = None
        raise ('Unknown network configuration: ', simulation_parameters['network'])

    # load model and push it to device
    map_location = torch.device(device)
    sd = torch.load(trained_net, map_location=map_location, weights_only=True)
    dnn.load_state_dict(sd)
    dnn.to(device)

    kalman = KalmanFilter(dim_x=2, dim_z=1)
    kalman.F = np.array([[1., 1.], [0., 1.]])
    kalman.H = np.array([[1., 0.]])
    kalman.P *= 10
    kalman.R = 5
    kalman.Q = Q_discrete_white_noise(dim=2, dt=1 / 30, var=0.5)
    kalman.x = np.array([0, 0.])

    list_predictions = []
    list_predictions_filtered = []
    list_eq = []
    list_targets = []
    list_head_rotation = []
    list_head_elevation = []
    list_head_roll = []
    list_face_distance = []
    list_error = []

    test_data_loader = DataLoader(dataset=dataset,
                                  batch_size=configuration['BATCH_SIZE'],
                                  num_workers=configuration['NUM_WORKERS'],
                                  persistent_workers=True,
                                  shuffle=False)

    idx = 0
    num_zeros = int(np.ceil(np.log10(len(dataset))) + 1)
    #with Timer('test_signals'):

    for image_left, image_right, target, metadata in test_data_loader:

        image_left = torch.unsqueeze(image_left, dim=0)
        image_right = torch.unsqueeze(image_right, dim=0)

        # load features to inference device (cpu/cuda)
        image_left = image_left.to(device)
        image_right = image_right.to(device)
        # load metadata to inference device (cpu/cuda)
        metadata = metadata.to(device)

        if simulation_parameters['use_metadata']:
            predicted = 2 * dnn.forward(image_left, image_right, metadata)
        else:
            predicted = 2 * dnn.forward(image_left, image_right)

        eq = apply_regression(predicted.cpu().detach().numpy()[0][0])

        kalman.predict()
        kalman.update(eq)

        list_predictions.append(predicted.cpu().detach().numpy()[0][0])
        list_predictions_filtered.append(kalman.x[0])
        list_eq.append(eq)
        list_targets.append(2*target[0].cpu().detach().numpy())
        list_head_rotation.append(metadata[0, 0].cpu().detach().numpy())
        list_head_elevation.append(metadata[0, 1].cpu().detach().numpy())
        list_head_roll.append(metadata[0, 2].cpu().detach().numpy())
        list_face_distance.append(metadata[0, 3].cpu().detach().numpy())

        sys.stdout.write("\r{0}>".format("=" * round(50*idx/len(dataset))))
        sys.stdout.flush()

        idx += 1

    # Write results to pandas table
    df = pd.DataFrame({
        'Target': list_targets,
        'Prediction': list_predictions
    })

    if save_result:
        df.to_csv(
            path_or_buf="easyTest.csv",
            index=False)

    print('\nPredictions:', 1000*np.std(np.array(list_predictions) - np.array(list_targets)))
    print('EQ:', 1000*np.std(np.array(list_eq) - np.array(list_targets)))
    print('EQ, Kalman filtered:', 1000 * np.std(np.array(list_predictions_filtered) - np.array(list_targets)))

    fig, ax = plt.subplots(1, 1)
    ax.plot(list_predictions, label='prediction')
    ax.plot(list_eq, label='eq')
    ax.plot(list_predictions_filtered, label='eq (kalman)')
    ax.plot(list_targets, label='target')
    if simulation_parameters['use_metadata']:
        ax.plot(list_head_rotation, label='head rotation')
        ax.plot(list_head_elevation, label='head elevation')
        ax.plot(list_head_roll, label='head roll')
        ax.plot(list_face_distance, label='face distance')
    ax.legend()
    plt.show()

    print("done.")
