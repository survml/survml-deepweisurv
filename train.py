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
BATCH_SIZE = 32


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


def neg_log_likelihood_loss(beta, eta, target):
    """

    :param beta:
    :param eta:
    :param target:
    :return:
    """
    status = target[0]
    time = target[1]
    beta = torch.mul(beta, SHAPE_SCALE)
    eta = torch.mul(eta, SCALE_SCALE)

    f = torch.log(torch.div(beta, eta)) \
        + (beta - 1) * torch.log(torch.div(time, eta)) \
        - torch.pow(torch.div(time, eta), beta)
    fu = torch.pow(torch.div(time, eta), beta)

    ll = torch.mul(status, f) + torch.mul((1 - status), (-fu))
    # print('time:', time, '\t', 'status:', status, '\t', 'beta:', beta, '\t', 'eta:', eta, '\t',
    # 'f:', f, '\t', 'fu:', fu, '\t', 'll:', - ll)

    return - ll


# load training set
ds_name = 'metabric'
df = pd.read_csv('data/survival-data-parsed/{}_parsed.csv'.format(ds_name))
time_name, status_name = get_time_status(ds_name)

X = df.drop(columns=[status_name, time_name])
y = df[[status_name, time_name]]

# normalization for features
X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_X_train, df_X_test, df_X_val, df_y_train, df_y_test, df_y_val = X_train, X_test, None, y_train, y_test, None

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

input_dim, output_dim = len(X.columns), 1

model = DeepWeiSurv(input_dim, output_dim)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
loss_list = []
model.eval()

train_ll_viz, val_ll_viz, shape_viz, scale_viz, ci_viz = [], [], [], [], []

# 'metabric': 100 iterations, lr = 0.01

global_step = 0

for t in range(100):
    ll = 0
    beta_ave, eta_ave = 0, 0
    torch.enable_grad()
    # optimizer.zero_grad()

    epoch_loss = 0
    for i, j in enumerate(X_train):
        global_step += 1

        beta_pred, eta_pred = model(torch.tensor(j).unsqueeze(0).float())
        beta_ave += beta_pred.squeeze(0)[0].item() / X_train.shape[0]
        eta_ave += eta_pred.squeeze(0)[0].item() / X_train.shape[0]

        loss = neg_log_likelihood_loss(beta_pred, eta_pred, y_train[i])
        loss /= X_train.shape[0]
        loss.backward()

        epoch_loss += loss.item()

        if global_step % BATCH_SIZE == 0:
            optimizer.step()
            optimizer.zero_grad()

    print("epoch:\t", t,
          "\t train loss:\t", "%.8f" % round(epoch_loss, 6),
          "\t shape:\t", "%.4f" % round(beta_ave * SHAPE_SCALE, 4),
          "\t scale:\t", "%.3f" % round(eta_ave * SCALE_SCALE, 3))

    train_ll_viz.append(loss.item())
    shape_viz.append(beta_ave * SHAPE_SCALE), scale_viz.append(eta_ave * SCALE_SCALE)

fig, axs = plt.subplots(3, figsize=[11, 20])
axs[0].plot(train_ll_viz)
axs[0].set_title("Average Train Loss")
axs[1].plot(shape_viz)
axs[1].set_title("Average Shape Parameter")
axs[2].plot(scale_viz)
axs[2].set_title("Average Scale Parameter")
fig.savefig("tests/img/dwsurv/dwsurv_train_{}_lr{}_{}.png".format(ds_name, learning_rate, uuid.uuid1()))
plt.show()

torch.save(model.state_dict(),
           'models/dwsurv/saved_models/dwsurv_{}_lr{}_bs{}_{}.pt'
           .format(ds_name, learning_rate, BATCH_SIZE, uuid.uuid1()))
