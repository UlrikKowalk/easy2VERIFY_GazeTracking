import argparse
import os

import torch
import torch.nn as nn
import yaml

from Networks.DNN_GADOAE_FiLM_multistage import DNN_GADOAE_FiLM_multistage
from Networks.DNN_GADOAE_FiLM_conv import DNN_GADOAE_FiLM_conv
from Networks.DNN_GADOAE_conv_res_FiLM import DNN_GADOAE_conv_res_FiLM
from Networks.DNN_GADOAE_conv_res_FiLM_phase import DNN_GADOAE_conv_res_FiLM_phase
from Networks.DNN_GADOAE_conv_res_FiLM_nonredunant import DNN_GADOAE_conv_res_FiLM_nonredundant
# from torch.utils.tensorboard import SummaryWriter
from Networks.DNN_GADOAE_max import DNN_GADOAE_max
from Networks.DNN_GADOAE_full import DNN_GADOAE_full
from Networks.DNN_GADOAE_parallel import DNN_GADOAE_parallel
from Networks.DNN_GADOAE_phase import DNN_GADOAE_phase
from Networks.DNN_GADOAE_conv import DNN_GADOAE_conv
from Core.Timer import Timer
from Dataset.Dataset_Training_offline import Dataset_Training_offline
from Core.Training import Training
from Core.Training_GADOAE_parallel import Training_GADOAE_parallel

# writer = SummaryWriter("runs/gcc")

with open('config_training_offline.yml') as config:
    configuration = yaml.safe_load(config)

training_parameters = configuration['training_parameters']
training_parameters['base_dir'] = os.getcwd()

def optimizer_to(optim, device):
    # suggested by user aaniin on the pytorch forum
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    print(f'DNN: {training_parameters["network"]}, loading model: {training_parameters["model"]}')

    with Timer("Inference"):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = "cpu"
        trained_net = f'{training_parameters["base_dir"]}/Trained_16k/{training_parameters["model"]}'
        print(f"Using device '{device}'.")

        if configuration['is_training']:

            dataset = Dataset_Training_offline(dataset=training_parameters['dataset'], device=device)

            if training_parameters['network'] == 'GADOAE_max':
                dnn = DNN_GADOAE_max(length_input_layer=dataset.get_length_feature(),
                                     length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_full':
                dnn = DNN_GADOAE_full(length_input_layer=dataset.get_length_feature(),
                                      length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_phase':
                dnn = DNN_GADOAE_phase(length_input_layer=dataset.get_length_feature(),
                                       length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_conv':
                dnn = DNN_GADOAE_conv(length_input_layer=dataset.get_length_feature(),
                                      length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_FiLM_redundant':
                dnn = DNN_GADOAE_conv_res_FiLM(length_input_layer=dataset.get_length_feature(),
                                      length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_FiLM_nonredundant':
                dnn = DNN_GADOAE_conv_res_FiLM_nonredundant(length_input_layer=dataset.get_length_feature(),
                                      length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_FiLM_phase':
                dnn = DNN_GADOAE_conv_res_FiLM_phase(length_input_layer=dataset.get_length_feature(),
                                      length_output_layer=dataset.get_num_classes())
            elif training_parameters['network'] == 'GADOAE_parallel':
                dnn = DNN_GADOAE_parallel(length_input_layer=dataset.get_length_feature(),
                                          length_output_layer=dataset.get_num_classes(),
                                          num_channels=training_parameters['num_channels'],
                                          device=device)
            elif training_parameters['network'] == 'GADOAE_FiLM':
                dnn = DNN_GADOAE_FiLM_conv(length_input_layer=dataset.get_length_feature(),
                                      length_output_layer=dataset.get_num_classes())
            else:
                dataset = None
                dnn = None
                raise ('Unknown network configuration: ', configuration['network'])

            optimiser = torch.optim.Adam(dnn.parameters(), lr=configuration['LEARNING_RATE'])

            if configuration['is_continuing']:
                sd = torch.load(trained_net, weights_only=False)
                dnn.load_state_dict(sd)
                op = torch.load(f'{trained_net}.opt', weights_only=False)
                optimiser.load_state_dict(op)

            dnn.to(device)
            optimizer_to(optimiser, device)

            loss_fn = nn.CrossEntropyLoss()

            if training_parameters['network'] == 'GADOAE_parallel':
                Trainer = Training_GADOAE_parallel(model=dnn, loss_fn=loss_fn,
                                                   optimiser=optimiser,
                                                   dataset=dataset,
                                                   batch_size=configuration['BATCH_SIZE'],
                                                   ratio=configuration['RATIO'],
                                                   device=device,
                                                   filename=trained_net,
                                                   num_workers=configuration['NUM_WORKERS'])
            else:
                Trainer = Training(model=dnn, loss_fn=loss_fn,
                                   optimiser=optimiser,
                                   dataset=dataset,
                                   batch_size=configuration['BATCH_SIZE'],
                                   ratio=configuration['RATIO'],
                                   device=device,
                                   filename=trained_net,
                                   num_workers=configuration['NUM_WORKERS'])

            with Timer("Training online"):
                # train model
                Trainer.train(epochs=configuration['EPOCHS'])

    # writer.close()
    print("done.")
