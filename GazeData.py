
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2


class GazeData(Dataset):

    def __init__(self, directory, device):

        self.directory = directory
        self.device = device
        self.dataframe = pd.DataFrame(pd.DataFrame({
                        'directory': [],
                        'filename': [],
                        'target': [],
                        'head_rotation': [],
                        'head_elevation': [],
                        'head_roll': [],
                        'face_distance': []
        }))

        for path in Path(self.directory).rglob('*.csv'):
            tmp_dataframe = pd.read_csv(path)
            num_elements = len(tmp_dataframe)
            tmp_dataframe['directory'] = [path.parent] * num_elements
            self.dataframe = self.dataframe._append(tmp_dataframe, ignore_index=True)

        self.length = len(self.dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        directory = self.dataframe['directory'][index]
        filename = self.dataframe['filename'][index]
        image = cv2.imread(f'{directory}/{filename}', cv2.IMREAD_GRAYSCALE).astype(np.float32)
        target = self.dataframe['target'][index].astype(np.float32)
        head_rotation = self.dataframe['head_rotation'][index]
        head_elevation = self.dataframe['head_elevation'][index]
        head_roll = self.dataframe['head_roll'][index]
        head_distance = self.dataframe['face_distance'][index]

        head_position = torch.tensor([head_rotation, head_elevation, head_roll, head_distance], dtype=torch.float32)

        return image, target, head_position