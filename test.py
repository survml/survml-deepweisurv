import uuid
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.utils.dataset import get_time_status

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

SHAPE_SCALE = 1
SCALE_SCALE = 100


class DeepWeiSurv(nn.Module):
    """
    Network architecture of DeepWeiSurv
    """

    def __init__(self, input, output):
        super(DeepWeiSurv, self).__init__()
        self.linear1 = nn.Linear(input, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.linear6 = nn.Linear(8, output)
        self.linear7 = nn.Linear(8, output)
        self.elu = nn.Softplus(output)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.bn1(x)
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = self.bn2(x)
        beta = self.elu(self.linear6(x))
        eta = self.elu(self.linear7(x))
        return beta, eta


ds_name = 'metabric'

# load training set
df = pd.read_csv('data/survival-data-parsed/{}_parsed.csv'.format(ds_name))

time_name, status_name = get_time_status(ds_name)

X = df.drop(columns=[status_name, time_name])
y = df[[status_name, time_name]]

# normalization for features
X = (X - X.min()) / (X.max() - X.min())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

df_X_train, df_X_test, df_X_val, df_y_train, df_y_test, df_y_val = X_train, X_test, None, y_train, y_test, None

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

input_dim, output_dim = len(X.columns), 1

model = DeepWeiSurv(input_dim, output_dim)
model.load_state_dict(
    torch.load('models/dwsurv/saved_models/dwsurv_metabric_lr0.01_bs32_b29314f0-491d-11ed-b3a4-acde48001122.pt'))
model.eval()

train_ll_viz, val_ll_viz, shape_viz, scale_viz, ci_viz = [], [], [], [], []

# 'metabric': 100 iterations, lr = 0.01

# -------------------------------------------------------------------------------------------------------------------- #
# ====================================== TESTING ===================================================================== #
# df_test = pd.read_excel('./data/test_metabric.xlsx')
# df_test = df_test.drop(['Unnamed: 0'], axis=1)
#
# Xt = df_test.drop(['time', 'status'], axis=1)
# Xt = (Xt - Xt.min()) / (Xt.max() - Xt.min())
# Xt = Xt.to_numpy()

model.eval()

beta_test, eta_test = [], []
for i, j in enumerate(X_test):
    beta_t, eta_t = model(torch.tensor(j).unsqueeze(0).float())
    beta_test.append(beta_t.squeeze(0)[0].item() * SHAPE_SCALE)
    eta_test.append(eta_t.squeeze(0)[0].item() * SCALE_SCALE)

df_temp = pd.DataFrame()
df_temp['time'] = df_y_test[time_name]
df_temp['status'] = df_y_test[status_name]
df_temp.reset_index(inplace=True)
df_temp = pd.concat([df_temp,
                     pd.DataFrame(beta_test, columns=['beta']),
                     pd.DataFrame(eta_test, columns=['eta'])], axis=1)
# df_temp['hr'] = (df_temp['beta'] / df_temp['eta']) * pow(100 / df_temp['eta'], df_temp['beta'] - 1)
# df_temp['hr'] = df_temp['eta'] * np.gamma(1 + 1 / df_temp['beta'])

mean_list = []
for m, n in enumerate(df_temp['eta']):
    # print(m, n)
    mean_list.append(n * math.gamma(1 + 1 / df_temp['beta'][m]))
df_temp = pd.concat([df_temp, pd.DataFrame(mean_list, columns=['mean'])], axis=1)
# df_temp['mean'].fillna(df_temp['mean'].mean(), inplace=True)
# ci = concordance_index(df_temp['time'], -df_temp['hr'], df_temp['status'])
ci = concordance_index(df_temp['time'], df_temp['mean'], df_temp['status'])

# ci metabric 0.6875860289057123

