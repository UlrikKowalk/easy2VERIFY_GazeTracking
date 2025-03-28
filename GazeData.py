
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import matplotlib.pyplot as plt


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

        # draw mask to focus on eye -> dimensions are known
        width = 100
        height = 100
        center = [50, 50]
        radius = 49
        self.mask = np.zeros(shape=(height, width), dtype=np.float32)
        for row in range(height):
            for col in range(width):
                if np.sqrt((center[0]-row)**2 + (center[1]-col)**2) <= radius:
                    self.mask[row, col] = 1.0
        self.mask = np.tile(A=self.mask, reps=(1, 2))

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

        # execute mask
        image *= self.mask

        head_position = torch.tensor([head_rotation, head_elevation, head_roll, head_distance], dtype=torch.float32)

        return image, target, head_position