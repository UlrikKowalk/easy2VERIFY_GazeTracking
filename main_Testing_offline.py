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
from Networks.DNN_GADOAE_max import DNN_GADOAE_max
from Networks.DNN_GADOAE_full import DNN_GADOAE_full
from Networks.DNN_GADOAE_phase import DNN_GADOAE_phase
from Networks.DNN_GADOAE_conv import DNN_GADOAE_conv
from Dataset.Dataset_Testing_offline import Dataset_Testing_offline
from Networks.DNN_GADOAE_conv_res_FiLM import DNN_GADOAE_FiLM_conv_redundant2
from Networks.DNN_GADOAE_parallel import DNN_GADOAE_parallel
from Algorithms.MUSIC import MUSIC
from Algorithms.SRP_PHAT import SRP_PHAT
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

    file_name = Evaluation.get_filename_offline_short(trained_net, simulation_parameters)
    print(f'\nTesting: {file_name}\n')

    dataset = Dataset_Testing_offline(parameters=simulation_parameters, device=device)

    if simulation_parameters['network'] == 'GADOAE_max':
        dnn = DNN_GADOAE_max(length_input_layer=dataset.get_length_feature(),
                             length_output_layer=dataset.get_num_classes())
    elif simulation_parameters['network'] == 'GADOAE_full':
        dnn = DNN_GADOAE_full(length_input_layer=dataset.get_length_feature(),
                              length_output_layer=dataset.get_num_classes())
    elif simulation_parameters['network'] == 'GADOAE_FiLM':
        dnn = DNN_GADOAE_FiLM_conv_redundant2(length_input_layer=dataset.get_length_feature(),
                                              length_output_layer=dataset.get_num_classes())
    elif simulation_parameters['network'] == 'GADOAE_phase':
        dnn = DNN_GADOAE_phase(length_input_layer=dataset.get_length_feature(),
                               length_output_layer=dataset.get_num_classes())
    elif simulation_parameters['network'] == 'GADOAE_conv':
        dnn = DNN_GADOAE_conv(length_input_layer=dataset.get_length_feature(),
                              length_output_layer=dataset.get_num_classes())
    elif simulation_parameters['network'] == 'GADOAE_parallel':
        dnn = DNN_GADOAE_parallel(length_input_layer=dataset.get_length_feature(),
                                  length_output_layer=dataset.get_num_classes(),
                                  num_channels=dataset.get_num_channels(),
                                  device=device)
    else:
        dnn = None
        raise ('Unknown network configuration: ', configuration['network'])

    # load model and push it to device
    map_location = torch.device(device)
    sd = torch.load(trained_net, map_location=map_location, weights_only=True)
    dnn.load_state_dict(sd)
    dnn.to(device)

    # add 'missing' information
    simulation_parameters['sampling_rate'] = dataset.get_sampling_rate()
    simulation_parameters['frame_length'] = dataset.get_frame_length()
    simulation_parameters['num_classes'] = dataset.get_num_classes()
    simulation_parameters['speed_of_sound'] = dataset.get_speed_of_sound()
    simulation_parameters['num_channels'] = dataset.get_num_channels()
    simulation_parameters['max_theta'] = dataset.get_max_theta()
    simulation_parameters['hop_size'] = dataset.get_hop_size()

    # dnn.set_hop_size(simulation_parameters['hop_size'])

    n_true = 0
    n_false = 0
    list_rt_60_testing = []
    list_snr_testing = []
    list_signal_type_testing = []
    list_ir_type_testing = []

    # print('Testing')

    list_predictions = []
    list_predictions_srpphat = []
    list_predictions_music = []
    list_targets = []
    list_var = []
    list_kalman = []
    list_error = []
    list_error_srpphat = []
    list_error_music = []
    list_vad_percentage = []
    list_vad_energy_percentage = []

    test_data_loader = DataLoader(dataset=dataset,
                                  batch_size=configuration['BATCH_SIZE'],
                                  num_workers=configuration['NUM_WORKERS'],
                                  persistent_workers=True,
                                  shuffle=False)

    idx = 0
    list_weird = []
    list_occ = [0] * dataset.get_num_classes()
    num_zeros = int(np.ceil(np.log10(len(dataset))) + 1)

    #with Timer('test_signals'):

    for features, target, coordinates, signal, voice_activity in test_data_loader:

        # make signal 2D again (reverse dataloader transformation)
        signal = signal[0, :, :]
        voice_activity = voice_activity[0]

        list_vad_energy_percentage.append((torch.mean(voice_activity) * 100).numpy())

        list_vad_percentage.append((torch.mean(voice_activity)*100).numpy())
        coordinates = coordinates[0, :, :].cpu().detach().numpy()

        predicted_srpphat = -255
        predicted_music = -255

        # dataset.save_audio(filename=f'testaudio.wav', audio=signal[0], normalize=False)

        # load features to inference device (cpu/cuda)
        features = features.to(device)
        # initialize SRP-PHAT
        srp_phat = SRP_PHAT(coordinates=coordinates,
                            parameters=simulation_parameters)
        # initialize MUSIC
        music = MUSIC(coordinates=coordinates,
                      parameters=simulation_parameters)

        predicted, _ = Evaluation.estimate_dnn(model=dnn, sample=features.squeeze(dim=0),
                                               voice_activity=voice_activity)

        # get statistics
        dist_min, dist_max, dist_mean = Evaluation.get_coordinate_distances(coordinates)
        # print(dist_min, dist_max, dist_mean)

        # predicted_srpphat, _ = Evaluation.estimate_srpphat(model=srp_phat, sample=signal,
        #                                                    voice_activity=voice_activity)
        #
        # predicted_music, _ = Evaluation.estimate_music(model=music, sample=signal,
        #                                                voice_activity=voice_activity)
        # predicted_neural_srp, _ = Evaluation.estimate_neural_srp(sample=signal, coordinates=coordinates, device=device)


        expected = int(target)

        # reset DOA shift (generalization towards unseen DOA's)
        if dataset.get_use_in_between_doas():
            print('using in-between')
            predicted -= 0.5
            predicted_srpphat -= 0.5
            predicted_music -= 0.5

        list_occ[expected] += 1

        ############################################################

        list_predictions.append(predicted)
        list_predictions_srpphat.append(predicted_srpphat)
        list_predictions_music.append(predicted_music)
        list_targets.append(expected)

        list_error.append(Evaluation.angular_error(expected, predicted,
                        dataset.get_num_classes()) / dataset.get_num_classes() * dataset.get_max_theta())
        list_error_srpphat.append(Evaluation.angular_error(expected, predicted_srpphat,
                        dataset.get_num_classes()) / dataset.get_num_classes() * dataset.get_max_theta())
        list_error_music.append(Evaluation.angular_error(expected, predicted_music,
                        dataset.get_num_classes()) / dataset.get_num_classes() * dataset.get_max_theta())

        sys.stdout.write("\r{0}>".format("=" * round(50*idx/len(dataset))))
        sys.stdout.flush()

        idx += 1

    # Write results to pandas table
    df = pd.DataFrame({
        'Target': list_targets,
        'Prediction': list_predictions,
        'Prediction_SRPPHAT': list_predictions_srpphat,
        'Prediction_MUSIC': list_predictions_music,
        'vad_percentage': list_vad_percentage,
        'vad_energy_percentage': list_vad_energy_percentage
    })

    if save_result:
        df.to_csv(
            path_or_buf=file_name,
            index=False)

    MAE_CNN = np.mean(list_error)
    MAE_SRPPHAT = np.mean(list_error_srpphat)
    MAE_MUSIC = np.mean(list_error_music)
    acc_model, acc_srpphat, acc_music = Evaluation.calculate_accuracy(df, simulation_parameters[
        'num_classes'])

    print(f'\nPercentage frames: {np.mean(list_vad_percentage)}')
    print(' ')
    print(f"DNN: MAE: {MAE_CNN} [{np.median(list_error)}], Accuracy: {acc_model}")
    print(f"SRP-PHAT: MAE: {MAE_SRPPHAT} [{np.median(list_error_srpphat)}], Accuracy: {acc_srpphat}")
    print(f"MUSIC: MAE: {MAE_MUSIC} [{np.median(list_error_music)}], Accuracy: {acc_music}")

    Evaluation.plot_error(df=df, num_classes=dataset.get_num_classes())
    print("done.")