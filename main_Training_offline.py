import argparse
import os

import torch
import torch.nn as nn
import yaml

from GazeData import GazeData
from Timer import Timer
from easyCNN_01 import easyCNN_01
from Training import Training

# writer = SummaryWriter("runs/gcc")

with open('config_training.yml') as config:
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

            dataset = GazeData(directory=training_parameters['dataset'], device=device)

            dnn = easyCNN_01()

            optimiser = torch.optim.Adam(dnn.parameters(), lr=configuration['LEARNING_RATE'])

            if configuration['is_continuing']:
                sd = torch.load(trained_net, weights_only=False)
                dnn.load_state_dict(sd)
                op = torch.load(f'{trained_net}.opt', weights_only=False)
                optimiser.load_state_dict(op)

            dnn.to(device)
            optimizer_to(optimiser, device)

            loss_fn = nn.CrossEntropyLoss()

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
