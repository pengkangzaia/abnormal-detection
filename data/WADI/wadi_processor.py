import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch
import os
import time
from datetime import datetime


class WADI:
    def get_columns(self):
        df = pd.read_csv(self.normal_data_path, skiprows=4, index_col=0)
        df = df.dropna(axis=1, how='all')
        df = df.drop(['Date', 'Time'], axis=1)
        col_names_res = []
        for i in df.columns:
            col_names_res.append(i.split('\\')[-1])
        return col_names_res

    def sliding_windows(self, data):
        x, y = [], []
        seq_length = self.window_size
        for i in range(len(data) - seq_length):
            _x = data[i:(i + seq_length)]
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def get_labels(self):
        testing_set = pd.read_csv(self.attack_data_path, usecols=['Date', 'Time'], nrows=self.read_rows)
        testing_set['Timestamp'] = testing_set['Date'] + " " + testing_set['Time']
        testing_set["Timestamp"] = pd.to_datetime(testing_set["Timestamp"], format="%m/%d/%Y %I:%M:%S.000 %p")
        testing_set["unix"] = testing_set["Timestamp"].astype(np.int64)
        abnormal_range = [['9/10/17 19:25:00', '9/10/17 19:50:16'], ['10/10/17 10:24:10', '10/10/17 10:34:00'],
                          ['10/10/17 10:55:00', '10/10/17 11:24:00'], ['10/10/17 11:30:40', '10/10/17 11:44:50'],
                          ['10/10/17 13:39:30', '10/10/17 13:50:40'], ['10/10/17 14:48:17', '10/10/17 14:59:55'],
                          ['10/10/17 17:40:00', '10/10/17 17:49:40'], ['10/10/17 10:55:00', '10/10/17 10:56:27'],
                          ['11/10/17 11:17:54', '11/10/17 11:31:20'], ['11/10/17 11:36:31', '11/10/17 11:47:00'],
                          ['11/10/17 11:59:00', '11/10/17 12:05:00'], ['11/10/17 12:07:30', '11/10/17 12:10:52'],
                          ['11/10/17 12:16:00', '11/10/17 12:25:36'], ['11/10/17 15:26:30', '11/10/17 15:37:00']]
        labels = np.zeros(testing_set.shape[0])
        for i in range(len(abnormal_range)):
            start = datetime.strptime(abnormal_range[i][0], "%d/%m/%y %H:%M:%S")
            # 手动设置时区为东八区
            start_timestamp = (int(time.mktime(start.timetuple())) + (8 * 60 * 60)) * (10 ** 9)

            end = datetime.strptime(abnormal_range[i][1], "%d/%m/%y %H:%M:%S")
            end_timestamp = (int(time.mktime(end.timetuple())) + (8 * 60 * 60)) * (10 ** 9)

            abnormal = testing_set[(testing_set['unix'] >= start_timestamp) & (testing_set['unix'] <= end_timestamp)]
            abnormal_idx = abnormal.index
            labels[abnormal_idx] = 1
        # 通过滑动窗口的y对应的label值
        labels = labels[self.window_size:]
        return labels

    def __init__(self, batch_size, window_size=12, read_rows=None):
        self.read_rows = read_rows
        # read normal data
        normal_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WADI_14days.csv')
        self.normal_data_path = normal_data_path
        normal = pd.read_csv(normal_data_path, skiprows=4, index_col=0, nrows=read_rows)
        normal = normal.dropna(axis=1, how='all')
        normal = normal.drop(['Date', 'Time'], axis=1)
        normal.columns = self.get_columns()
        normal.fillna(0, inplace=True)
        normal = normal.astype(float)
        sc = preprocessing.MinMaxScaler()
        normal = sc.fit_transform(normal)
        normal = pd.DataFrame(normal)

        # read attack data
        attack_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WADI_attackdata.csv')
        self.attack_data_path = attack_data_path
        attack = pd.read_csv(attack_data_path, index_col=0, nrows=read_rows)
        attack = attack.dropna(axis=1, how='all')
        attack = attack.drop(['Date', 'Time'], axis=1)
        attack.columns = self.get_columns()
        attack.fillna(0, inplace=True)
        attack = attack.astype(float)
        attack = sc.fit_transform(attack)
        attack = pd.DataFrame(attack)

        # 窗口化
        windows_normal = normal.values[
            np.arange(window_size)[None, :] + np.arange(normal.shape[0] - window_size)[:, None]]
        windows_attack = attack.values[
            np.arange(window_size)[None, :] + np.arange(attack.shape[0] - window_size)[:, None]]

        windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
        windows_normal_val = windows_normal[
                             int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        self.train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(
                ([windows_normal_train.shape[0], windows_normal_train.shape[1], windows_normal_train.shape[2]]))
        ), batch_size=batch_size, shuffle=False, num_workers=0)

        self.val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(
                ([windows_normal_val.shape[0], windows_normal_train.shape[1], windows_normal_train.shape[2]]))
        ), batch_size=batch_size, shuffle=False, num_workers=0)

        self.test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(
                ([windows_attack.shape[0], windows_attack.shape[1], windows_attack.shape[2]]))
        ), batch_size=batch_size, shuffle=False, num_workers=0)

        self.input_feature_dim = windows_normal.shape[2]
        self.window_size = window_size

    def get_dataloader(self):
        return self.train_loader, self.val_loader, self.test_loader


