import math
import time
import h5py
import logging
import argparse
import numpy as np
import os, sys, json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader




def _dataset_to_tensor(dset, mask=None, dtype=np.int64):
    arr = np.asarray(dset, dtype=dtype)
    if mask is not None:
        arr = arr[mask]
    if dtype == np.float32:
        tensor = torch.FloatTensor(arr)
    else:
        tensor = torch.LongTensor(arr)
    return tensor



class OSICDataset(Dataset):
    def __init__(self,
                 train_csv,
                 CT_h5,
                 mode):

        self.train_csv = train_csv
        self.CT_h5 = CT_h5
        self.mode = mode
        np.random.seed()
        print('Reading training data into memory')
        self.data_len = len(CT_h5['CT'])


        col_sex = self.train_csv["Sex"]
        data_sex = np.array(col_sex)
        data_sex_re = data_sex.reshape(data_len,1)

        col_age = self.train_csv["Age"]
        data_age = np.array(col_age)
        data_age_re = data_sex.reshape(data_len,1)

        col_smoke = self.train_csv["SmokingStatus"]
        data_smoke = np.array(col_smoke)
        data_smoke_re = data_sex.reshape(data_len,1)

        data_other_1 = np.hstack((data_sex_re,data_age_re))
        data_other = np.hstack((data_other_1,data_smoke_re))

        self.CT_image_tensor = _dataset_to_tensor(CT_h5['CT'])
        self.data_other_tensor = _dataset_to_tensor(other)


        col_week = self.train_csv["Weeks"]
        data_week = np.array(col_week) + 12  #first week is -12
        if self.mode == 'train':
            data_week_re = data_week.reshape(data_len,9)

            col_part_fvc = self.train_csv["FVC"]
            data_fvc = np.array(col_part_fvc)
            data_fvc_re = data_fvc.reshape(data_len,9)

            FVC_input = np.zeros((data_len,146))   #146 = 133+12
            FVC_label = np.zeros((data_len,146))   #146 = 133+12

            for x_index in range(data_len):
                FVC_input[x_index][data_week_re[x_index][0]] = data_fvc_re[x_index][0]
                for y_index in range(9):
                    FVC_input[x_index][data_week_re[x_index][y_index]] = data_fvc_re[x_index][y_index]

            self.FVC_input_tensor = _dataset_to_tensor(FVC_input)
            self.FVC_label_tensor = _dataset_to_tensor(FVC_label)

        elif self.mode == 'eval':
            data_week_re = data_week.reshape(data_len,1)
            col_part_fvc = self.train_csv["FVC"]
            data_fvc = np.array(col_part_fvc)
            data_fvc_re = data_fvc.reshape(data_len,1)

            FVC_input = np.zeros((data_len,146))   #146 = 133+12
            for x_index in range(data_len):
                FVC_input[x_index][data_week_re[x_index][0]] = data_fvc_re[x_index][0]

            self.FVC_input_tensor = _dataset_to_tensor(FVC_input)



    def __getitem__(self, index):
        if self.mode = 'train':
            CT = self.CT_image_tensor[index]
            other_data = self.data_other_tensor[index]
            FVC_input = self.FVC_input_tensor[index]
            FVC_label = self.FVC_label_tensor[index]
            return  CT , other_data, FVC_input,FVC_label
        elif self.mode = 'eval':
            CT = self.CT_image_tensor[index]
            other_data = self.data_other_tensor[index]
            FVC_input = self.FVC_input_tensor[index]
            return  CT , other_data, FVC_input

    def __len__(self):
        return self.data_len


class EqaDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'train_csv' not in kwargs:
            raise ValueError('Must give train_csv')
        if 'CT_h5' not in kwargs:
            raise ValueError('Must give CT_h5')
        if 'mode' not in kwargs:
            raise ValueError('Must give mode')


        train_csv_path = kwargs.pop('train_csv')
        CT_path = kwargs.pop('CT_h5')
        mode = kwargs.pop('mode')

        train_csv = pd.read_csv(train_csv_path)
        print('Reading csv from ', train_csv_path)

        CT_h5 = h5py.File(CT_path, 'r')
        print('Reading CT from ', CT_path)

        self.dataset = EqaDataset(
            train_csv,
            CT_h5,
            mode
            )

        super(EqaDataLoader, self).__init__(self.dataset, **kwargs)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-CT_h5', default='../data/CT_h5')
    parser.add_argument('-train_csv', default='../data/train_csv')
    parser.add_argument('-mode', default='train',choices=['train','eval'])
    parser.add_argument('-batch_size', default=2, type=int)
    args = parser.parse_args()

    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        exit()

    train_loader_kwargs = {
        'CT_h5': args.CT_h5,
        'train_csv': args.train_csv,
        'mode': args.mode,
        'batch_size': args.batch_size,
    }

    train_loader = EqaDataLoader(**train_loader_kwargs)



