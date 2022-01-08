import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch
import os


class SWaT:
    def __init__(self, batch_size, window_size=12, read_rows=None):
        # Read normal data
        normal_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SWaT_Dataset_Normal_v1.csv')
        attack_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SWaT_Dataset_Attack_v0.csv')
        normal = pd.read_csv(normal_data_path, nrows=read_rows)
        normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
        # Transform all columns into float64
        for i in list(normal):
            normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
        normal = normal.astype(float)

        # 数据预处理
        min_max_scaler = preprocessing.MinMaxScaler()
        x = normal.values
        x_scaled = min_max_scaler.fit_transform(x)
        normal = pd.DataFrame(x_scaled)

        # Read attack data
        attack = pd.read_csv(attack_data_path, sep=";", nrows=read_rows)
        self.attack_labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
        self.attack_labels = np.array(self.attack_labels[window_size:])
        attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)

        # Transform all columns into float64
        for i in list(attack):
            attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype(float)

        x = attack.values
        x_scaled = min_max_scaler.transform(x)
        attack = pd.DataFrame(x_scaled)

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
