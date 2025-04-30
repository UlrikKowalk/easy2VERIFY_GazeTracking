import os
import random

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler


writer = SummaryWriter("runs/gcc")


class Training:

    def __init__(self, model, loss_fn, optimiser, dataset, batch_size, ratio, device, filename, num_workers, use_metadata=False):

        self.train_data_loader = None
        self.val_data_loader = None
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.filename = filename
        self.use_metadata = use_metadata

        n_total = len(dataset)
        n_train = int(n_total * ratio)
        n_val = n_total - n_train
        train_data, val_data = torch.utils.data.random_split(dataset=dataset, lengths=[n_train, n_val])
        self.train_data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        self.val_data_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # def set_all_seeds(self, seed):
    #     random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    #     np.random.seed(seed)
    #     rand_gen = torch.Generator()
    #     rand_gen.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     return rand_gen

    def train(self, epochs):

        loss_list = []
        loss_val_list = []

        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")

            # Train
            loss = self.train_single_epoch()
            loss_list.append(loss)

            # Validate
            loss_val = self.validate_single_epoch()
            loss_val_list.append(loss_val)

            torch.save(self.model.state_dict(), self.filename)
            torch.save(self.optimiser.state_dict(), f'{self.filename}.opt')

            writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={'training': loss, 'validation': loss_val},
                global_step=i)
            writer.close()

            print("---------------------------")
        print("Finished training")

        return loss_list, loss_val_list

    def train_single_epoch(self):

        losses = []

        for image_left, image_right, target, metadata in self.train_data_loader:

            image_left = torch.unsqueeze(image_left, dim=1)
            image_right = torch.unsqueeze(image_right, dim=1)
            image_left = torch.unsqueeze(image_left, dim=1)
            image_right = torch.unsqueeze(image_right, dim=1)
            target = torch.unsqueeze(target, dim=1)

            image_left, image_right, target, metadata = (image_left.to(self.device),
                                                              image_right.to(self.device),
                                                              target.to(self.device),
                                                              metadata.to(self.device))

            self.model.train()
            self.optimiser.zero_grad()

            if self.use_metadata:
                prediction = self.model(image_left, image_right, metadata)
            else:
                prediction = self.model(image_left, image_right)

            # calculate loss
            loss = self.loss_fn(prediction, target)
            losses.append(loss.item())

            # backpropagate error and update weights
            loss.backward()
            self.optimiser.step()

        loss = np.mean(np.array(losses))
        print(f"loss: {loss.item()}")
        return loss.item()

    def validate_single_epoch(self):

        for image_left, image_right, target, metadata in self.train_data_loader:

            image_left = torch.unsqueeze(image_left, dim=1)
            image_right = torch.unsqueeze(image_right, dim=1)
            image_left = torch.unsqueeze(image_left, dim=1)
            image_right = torch.unsqueeze(image_right, dim=1)
            target = torch.unsqueeze(target, dim=1)

            # bulk_head_position = torch.unsqueeze(bulk_head_position, dim=1)
            image_left, image_right, target, metadata = (image_left.to(self.device),
                                                              image_right.to(self.device),
                                                              target.to(self.device),
                                                              metadata.to(self.device))

            self.model.eval()
            with torch.no_grad():
                # evaluate
                if self.use_metadata:
                    prediction = self.model(image_left, image_right, metadata)
                else:
                    prediction = self.model(image_left, image_right)
                # calculate loss
                loss = self.loss_fn(prediction, target)

        print(f"val:  {loss.item()}")
        return loss.item()
