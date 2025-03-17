import os
import random

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
writer = SummaryWriter("runs/gcc")


class Training:

    def __init__(self, model, loss_fn, optimiser, dataset, batch_size, ratio, device, filename, num_workers):

        self.train_data_loader = None
        self.val_data_loader = None
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.filename = filename
        self.num_classes = dataset.get_num_classes()

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

        list_snr = []
        list_rt_60 = []
        list_signal_type = []
        list_ir_type = []

        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")

            # Train
            loss, rt_60, snr, signal_type, ir_type = self.train_single_epoch()
            loss_list.append(loss)

            list_rt_60.extend(rt_60)
            list_snr.extend(snr)
            list_signal_type.extend(signal_type)
            list_ir_type.extend(ir_type)

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

        return loss_list, loss_val_list, list_rt_60, list_snr, list_signal_type, list_ir_type

    def train_single_epoch(self):

        losses = []
        list_rt_60 = []
        list_snr = []
        list_signal_type = []
        list_ir_type = []

        for bulk_sample, bulk_target, bulk_rotation, bulk_elevation, bulk_tilt in self.train_data_loader:

            sample, label, rotation, elevation, tilt = (bulk_sample.to(self.device), bulk_target.to(self.device), 
                                                        bulk_rotation.to(self.device), bulk_elevation.to(self.device),
                                                        bulk_tilt.to(self.device))

            print(bulk_sample.shape, bulk_target.shape)


            num_frames = int(sample.shape[0] * sample.shape[1])
            sample_reshaped = torch.zeros(size=(num_frames, sample.shape[2]), device=self.device)
            label_reshaped = torch.zeros(size=(num_frames, self.num_classes), device=self.device, dtype=torch.float32)

            idx = 0
            for frame, tmp_label in zip(sample, label):

                frames_per_sample = 1#label.shape[1]
                sample_reshaped[int(idx*frames_per_sample):int((idx+1)*frames_per_sample), :] = frame
                label_reshaped[int(idx*frames_per_sample):int((idx+1)*frames_per_sample), :] = \
                    torch.nn.functional.one_hot(tmp_label.long(), num_classes=self.num_classes)

                idx += 1

            self.optimiser.zero_grad()

            prediction = self.model(sample_reshaped)

            # calculate loss
            loss = self.loss_fn(prediction, label_reshaped)
            losses.append(loss.item())

            # backpropagate error and update weights
            loss.backward()
            self.optimiser.step()

        loss = np.mean(np.array(losses))
        print(f"loss: {loss.item()}")
        return loss.item(), list_rt_60, list_snr, list_signal_type, list_ir_type

    def validate_single_epoch(self):

        losses = []
        for bulk_sample, bulk_target in self.val_data_loader:

            sample, label = bulk_sample.to(self.device), bulk_target.to(self.device)
            num_frames = int(sample.shape[0] * sample.shape[1])
            sample_reshaped = torch.zeros(size=(num_frames, sample.shape[2]), device=self.device)
            label_reshaped = torch.zeros(size=(num_frames, self.num_classes), device=self.device,
                                         dtype=torch.float32)

            idx = 0
            for frame, tmp_label in zip(sample, label):
                frames_per_sample = 1#label.shape[1]
                sample_reshaped[int(idx * frames_per_sample):int((idx + 1) * frames_per_sample), :] = frame
                label_reshaped[int(idx * frames_per_sample):int((idx + 1) * frames_per_sample), :] = \
                    torch.nn.functional.one_hot(tmp_label.long(), num_classes=self.num_classes)

                idx += 1

            self.model.eval()
            with torch.no_grad():
                # evaluate
                prediction = self.model(sample_reshaped)
                # calculate loss
                loss = self.loss_fn(prediction, label_reshaped)

        print(f"val:  {loss.item()}")
        return loss.item()
