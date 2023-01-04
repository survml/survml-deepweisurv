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


class Net(nn.Module):
    """
    Network architecture of DeepWeiSurv
    """

    def __init__(self, input, output):
        super(Net, self).__init__()
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


class Data(Dataset):
    def __init__(self):
        self.X = torch.from_numpy(X_train)
        self.Y = torch.from_numpy(y_train)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


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


ds_name = 'metabric'

# load training set
df = pd.read_csv('data/survival-data-parsed/{}_parsed.csv'.format(ds_name))

time_name, status_name = get_time_status(ds_name)

X = df.drop(columns=[status_name, time_name])
y = df[[status_name, time_name]]

# normalization for features
X = (X - X.min()) / (X.max() - X.min())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

df_X_train, df_X_test, df_X_val, df_y_train, df_y_test, df_y_val = X_train, X_test, X_val, y_train, y_test, y_val

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
# X.shape, Y.shape
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()

data = Data()
loader = DataLoader(dataset=data, batch_size=64)

input_dim, output_dim = len(X.columns), 1

model = Net(input_dim, output_dim)
# print(clf.parameters)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
loss_list = []
model.eval()

train_ll_viz, val_ll_viz, shape_viz, scale_viz, ci_viz = [], [], [], [], []

# 'metabric': 100 iterations, lr = 0.01

for t in range(300):
    ll = 0
    beta_ave, eta_ave = 0, 0
    # for i, j in enumerate(loader):
    for i, j in enumerate(X_train):
        beta_pred, eta_pred = model(torch.tensor(j).unsqueeze(0).float())
        beta_ave += beta_pred.squeeze(0)[0].item() / X_train.shape[0]
        eta_ave += eta_pred.squeeze(0)[0].item() / X_train.shape[0]

        loss = neg_log_likelihood_loss(beta_pred, eta_pred, y_train[i])
        loss /= X_train.shape[0]
        # print(beta_pred, eta_pred, data.Y[i], loss.item())
        loss.backward()

    beta_val, eta_val = [], []
    vloss_total = 0
    for i, j in enumerate(X_val):
        beta_t, eta_t = model(torch.tensor(j).unsqueeze(0).float())
        vloss = neg_log_likelihood_loss(beta_t, eta_t, y_val[i])
        vloss /= X_val.shape[0]
        vloss_total += vloss.item()

        beta_val.append(beta_t.squeeze(0)[0].item() * SHAPE_SCALE)
        eta_val.append(eta_t.squeeze(0)[0].item() * SCALE_SCALE)

    df_temp = pd.DataFrame()
    df_temp['time'] = df_y_val[time_name]
    df_temp['status'] = df_y_val[status_name]
    df_temp.reset_index(inplace=True)
    df_temp = pd.concat([df_temp, pd.DataFrame(beta_val, columns=['beta']),
                         pd.DataFrame(eta_val, columns=['eta'])], axis=1)
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

    # ll = torch.div(ll, data.X.size(0))
    # loss = -ll
    print("epoch:\t", t,
          "\t train loss:\t", "%.8f" % round(loss.item(), 6),
          "\t valid loss:\t", "%.8f" % round(vloss.item(), 6),
          "\t shape:\t", "%.4f" % round(beta_ave * SHAPE_SCALE, 4),
          "\t scale:\t", "%.3f" % round(eta_ave * SCALE_SCALE, 3),
          "\t concordance index:\t", "%.8f" % round(ci, 8))

    train_ll_viz.append(loss.item()), val_ll_viz.append(vloss.item())
    shape_viz.append(beta_ave * SHAPE_SCALE), scale_viz.append(eta_ave * SCALE_SCALE)
    ci_viz.append(ci)

    optimizer.step()
    # scheduler.step()
    optimizer.zero_grad()
    # with torch.no_grad():
    #     for param in clf.parameters():
    #         print(param.grad)
    #         param -= learning_rate * param.grad

fig, axs = plt.subplots(4, figsize=[11, 20])
axs[0].plot(train_ll_viz)
axs[0].set_title("Average Train Loss")
axs[0].plot(val_ll_viz)
axs[0].set_title("Average Validation Loss")
axs[1].plot(shape_viz)
axs[1].set_title("Average Shape Parameter")
axs[2].plot(scale_viz)
axs[2].set_title("Average Scale Parameter")
axs[3].plot(ci_viz)
axs[3].set_title("Concordance Index")
fig.savefig("tests/img/dwsurv/dwsurv_{}_lr{}_{}.png".format(ds_name, learning_rate, uuid.uuid1()))
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# ====================================== TESTING ===================================================================== #
# df_test = pd.read_excel('./data/test_metabric.xlsx')
# df_test = df_test.drop(['Unnamed: 0'], axis=1)
#
# Xt = df_test.drop(['time', 'status'], axis=1)
# Xt = (Xt - Xt.min()) / (Xt.max() - Xt.min())
# Xt = Xt.to_numpy()
#
# model.eval()
#
# beta_test, eta_test = [], []
# for i, j in enumerate(Xt):
#     # print(i, j)
#     beta_t, eta_t = model(torch.tensor(j).unsqueeze(0).float())
#     beta_test.append(beta_t.squeeze(0)[0].item() * SHAPE_SCALE)
#     eta_test.append(eta_t.squeeze(0)[0].item() * SCALE_SCALE)
#
# # results_test = pd.DataFrame()
# results_test = pd.concat([df_test, pd.DataFrame(beta_test, columns=['beta']),
#                           pd.DataFrame(eta_test, columns=['eta'])], axis=1)
#
# results_test.to_excel("./results/results_test_metabric_{}.xlsx".format(uuid.uuid1()))
#
# # results_test['hr'] = (results_test['beta'] / results_test['eta']) * pow(100 / results_test['eta'],
# #                                                                         results_test['beta'] - 1)
# # results_test['hr'] = results_test['eta'] * (1 + 1 / results_test['beta'])
# mean_list = []
# for m, n in enumerate(results_test['eta']):
#     # print(m, n)
#     mean_list.append(n * math.gamma(1 + 1 / results_test['beta'][m]))
# results_test = pd.concat([results_test, pd.DataFrame(mean_list, columns=['mean'])], axis=1)
#
# ci = concordance_index(results_test['time'], results_test['hr'], results_test['status'])
# # ci = concordance_index(results_test['time'], -results_test['hr'], results_test['status'])
