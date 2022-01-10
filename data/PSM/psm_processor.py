import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch
import os


class PSM:
    def __init__(self, batch_size, window_size=12, read_rows=None):
        # Read normal data
        normal_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv')
        attack_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.csv')
        attack_label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_label.csv')
        normal = pd.read_csv(normal_data_path, nrows=read_rows)
        # Transform all columns into float64
        normal.fillna(0, inplace=True)
        normal = normal.astype(float)

        # 数据预处理
        min_max_scaler = preprocessing.MinMaxScaler()
        x = normal.values
        x_scaled = min_max_scaler.fit_transform(x)
        normal = pd.DataFrame(x_scaled)

        # Read attack data
        attack = pd.read_csv(attack_data_path, nrows=read_rows)
        attack = attack.astype(float)

        x = attack.values
        x_scaled = min_max_scaler.transform(x)
        attack = pd.DataFrame(x_scaled)

        self.attack_labels = pd.read_csv(attack_label_path, nrows=read_rows, usecols=['label'])
        self.attack_labels = np.array(self.attack_labels[window_size:])

        # 窗口化数据
        # np.arange(window_size)[None, :] 1*12 (0,1,2,3,4,5,6,7,8,9,10,11)一行12列
        # np.arange(normal.shape[0] - window_size)[:, None] (1000-12)*1 (0,1,2,3,4,5...) 988列，每列递增
        # np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size)[:, None] (1000-12)*12
        windows_normal = normal.values[
            np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size)[:, None]]
        windows_attack = attack.values[
            np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size)[:, None]]

        windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
        windows_normal_val = windows_normal[
                             int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        self.train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(
                ([windows_normal_train.shape[0], windows_normal_train.shape[1] * windows_normal_train.shape[2]]))
        ), batch_size=batch_size, shuffle=False, num_workers=0)

        self.val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(
                ([windows_normal_val.shape[0], windows_normal_train.shape[1] * windows_normal_train.shape[2]]))
        ), batch_size=batch_size, shuffle=False, num_workers=0)

        self.test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(
                ([windows_attack.shape[0], windows_attack.shape[1] * windows_attack.shape[2]]))
        ), batch_size=batch_size, shuffle=False, num_workers=0)

        self.input_feature_dim = windows_normal.shape[2]
        self.window_size = window_size

    def get_dataloader(self):
        return self.train_loader, self.val_loader, self.test_loader
