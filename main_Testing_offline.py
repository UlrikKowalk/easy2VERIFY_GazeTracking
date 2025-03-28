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
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import Core.Evaluation as Evaluation
from GazeData import GazeData
from easyCNN_01 import easyCNN_01
from Core.Timer import Timer
from random import Random

with open('config_testing_offline.yml') as config:
    configuration = yaml.safe_load(config)

save_result = configuration['save_result']
# evaluation_parameters = configuration['evaluation_parameters']
simulation_parameters = configuration['simulation_parameters']
simulation_parameters['base_dir'] = os.getcwd()
# simulation_parameters['noise_sampled_dir'] = '../NoiseLibrary/noise_sampled_05'
# simulation_parameters['mic_array'] = './array_viwers_05.mat'


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

    dataset = GazeData(directory=simulation_parameters["directory"], device=device)

    dnn = easyCNN_01()

    # load model and push it to device
    map_location = torch.device(device)
    sd = torch.load(trained_net, map_location=map_location, weights_only=True)
    dnn.load_state_dict(sd)
    dnn.to(device)

    list_predictions = []
    list_targets = []
    list_error = []

    test_data_loader = DataLoader(dataset=dataset,
                                  batch_size=configuration['BATCH_SIZE'],
                                  num_workers=configuration['NUM_WORKERS'],
                                  persistent_workers=True,
                                  shuffle=False)

    idx = 0
    num_zeros = int(np.ceil(np.log10(len(dataset))) + 1)
    #with Timer('test_signals'):

    for feature, target, metadata in test_data_loader:

        # load features to inference device (cpu/cuda)
        feature = feature.to(device)
        # load metadata to inference device (cpu/cuda)
        metadata = metadata.to(device)

        predicted, _ = Evaluation.estimate_easyCNN(model=dnn, feature=feature, metadata=metadata)
        expected = int(target)

        list_predictions.append(predicted)
        list_targets.append(expected)

        list_error.append(Evaluation.angular_error(expected, predicted,
                        dataset.get_num_classes()) / dataset.get_num_classes() * dataset.get_max_theta())

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

    MAE_CNN = np.mean(list_error)
    acc_model = Evaluation.calculate_accuracy(df, simulation_parameters[
        'num_classes'])
    print(f"DNN: MAE: {MAE_CNN} [{np.median(list_error)}], Accuracy: {acc_model}")

    Evaluation.plot_error(df=df, num_classes=dataset.get_num_classes())
    print("done.")