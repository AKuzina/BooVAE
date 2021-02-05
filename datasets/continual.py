import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X, y, transforms=None,
                 all_tasks=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]):
        super(CustomDataset, self).__init__()
        self.all_tasks = all_tasks
        self.current_task = None
        self.y = y.clone()
        self.X = X.clone()

        self.prepare_data()

        self.transform = transforms
        # self.tensors = [self.X, self.y]

    def prepare_data(self):
        unique_labels = np.unique(self.all_tasks)
        # this will be changed depending on the task
        self.X = self.X[np.isin(self.y, unique_labels)]
        self.y = self.y[np.isin(self.y, unique_labels)]

        # this will no be changed
        self.all_y = self.y.clone()
        self.all_x = self.X.clone()

    def __getitem__(self, item):
        img = self.X[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[item]

    def __len__(self):
        return self.y.shape[0]

    def set_task(self, task_id):
        self.current_task = task_id
        # x_train = x_train[]
        idx = np.isin(self.all_y, self.all_tasks[task_id])
        self.y = self.all_y[idx]
        self.X = self.all_x[idx]

    def add_coreset(self, size=30):
        if size > 0:
            # select random images from previous tasks
            core_x = []
            core_y = []
            for i in range(self.current_task):
                tsk_id = self.all_y == i
                core_id = np.random.randint(0, torch.sum(tsk_id), size)
                core_x.append(self.all_x[tsk_id][core_id])
                core_y.append(self.all_y[tsk_id][core_id])
            self.add_data(torch.cat(core_x), torch.cat(core_y))
            print('Coreset added for {} class(es).'.format(len(core_x)))

    def add_data(self, X, y):
        # add selected images to the dataset
        im = self.X.shape[0]
        self.X = torch.cat([self.X, X])
        self.y = torch.cat([self.y, y])
        print('{} images added in total.'.format(self.X.shape[0] - im))

def select_subsample_boo(args, train_dset):
    K = 1000
    if args.incremental:
        X = []
        for i in range(args.max_tasks):
            train_dset.set_task(i)
            temp_loader = DataLoader(train_dset, batch_size=K, shuffle=True)
            X.append(iter(temp_loader).next()[0])
    else:
        temp_loader = DataLoader(train_dset, batch_size=K, shuffle=True)
        X = [iter(temp_loader).next()[0]]
    return X
