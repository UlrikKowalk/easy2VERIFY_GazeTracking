
from pathlib import Path
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
                        'head_tilt': []
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
        image = cv2.imread(f'{directory}/{filename}')
        target = self.dataframe['target'][index]
        head_rotation = self.dataframe['head_rotation'][index]
        head_elevation = self.dataframe['head_elevation'][index]
        head_tilt = self.dataframe['head_tilt'][index]

        return image, target, head_rotation, head_elevation, head_tilt