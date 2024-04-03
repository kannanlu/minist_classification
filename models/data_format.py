# coding=UTF-8
"""@Description: data loader for the model
    @Author: Kannan Lu, lukannan@link.cuhk.edu.hk
    @Date: 2024/04/02
"""
from torch.utils import data


class MNISTDataset(data.Dataset):
    """data set formatter 
    """

    def __init__(self, x) -> None:
        super().__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data_point = self.x[index][0]
        data_label = self.x[index][1]
        return data_point, data_label
