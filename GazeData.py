
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import matplotlib.pyplot as plt


class GazeData(Dataset):

    def __init__(self, directory, device):

        self.internal_index = 0
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
        self.width = 100
        self.height = 100

        # draw mask to focus on eye -> dimensions are known
        center = [int(self.width/2), int(self.height/2)]
        radius = int(self.width/2)
        self.mask = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        for row in range(self.height):
            for col in range(self.width):
                if np.sqrt((center[0]-row)**2 + (center[1]-col)**2) <= radius:
                    self.mask[row, col] = 1.0
        # self.mask = np.tile(A=self.mask, reps=(1, 2))

    def __len__(self):
        return self.length

    def set_length(self, length):
        self.length = length

    def __getitem__(self, index):

        index = self.internal_index
        self.internal_index += 1

        directory = self.dataframe['directory'][index]
        filename = self.dataframe['filename'][index]
        image = cv2.imread(f'{directory}/{filename}', cv2.IMREAD_GRAYSCALE).astype(np.float32)
        target = self.dataframe['target'][index]
        head_rotation = self.dataframe['head_rotation'][index]
        head_elevation = self.dataframe['head_elevation'][index]
        head_roll = self.dataframe['head_roll'][index]
        face_distance = self.dataframe['face_distance'][index]

        # split eyes and execute mask
        image_left = image[:, :self.width] * self.mask
        image_right = image[:, self.width:] * self.mask

        image_left = image_left[20:-20, 20:-20]
        image_right = image_right[20:-20, 20:-20]

        # invert image
        # image_left = 512 - image_left
        # image_right = 512 - image_right
        #
        # plt.imshow(image_left)
        # plt.show()

        head_rotation = head_rotation / 180
        head_elevation = head_elevation / 180
        head_roll = head_roll / 180
        face_distance /= 300

        head_position = torch.tensor([head_rotation, head_elevation, head_roll, face_distance], dtype=torch.float32)

        #condition target values to be on interval [-1,1]
        target = 0.5*torch.tensor(target / torch.pi, dtype=torch.float32)

        return image_left, image_right, target, head_position