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

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from models.utils.dataset import get_time_status

SHAPE_SCALE = 1
SCALE_SCALE = 100
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 30
DATASET = 'metabric'


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

    return - ll


def validation(model2valid, valid_inputs, valid_targets):
    """

    :param model2valid:
    :param valid_inputs:
    :param valid_targets:
    :return:
    """
    shape_list, scale_list = [], []
    valid_loss = 0
    for index, input_features in enumerate(valid_inputs):
        shape, scale = model2valid(torch.tensor(input_features).unsqueeze(0).float())
        step_loss = neg_log_likelihood_loss(shape, scale, valid_targets[index])
        step_loss /= valid_inputs.shape[0]
        valid_loss += step_loss.item()

        shape_list.append(shape.squeeze(0)[0].item() * SHAPE_SCALE)
        scale_list.append(scale.squeeze(0)[0].item() * SCALE_SCALE)

    valid_results = pd.DataFrame(valid_targets, columns=['status', 'time'])  # status; time
    valid_mean = []
    for index, (shape, scale) in enumerate(zip(shape_list, scale_list)):
        valid_mean.append(scale * math.gamma(1 + 1 / shape))
    valid_ci = concordance_index(valid_results['time'], pd.DataFrame(valid_mean), valid_results['status'])

    return valid_loss, valid_ci


if __name__ == "__main__":

    # load training set
    df = pd.read_csv('data/survival-data-parsed/{}_parsed.csv'.format(DATASET))
    time_name, status_name = get_time_status(DATASET)

    X = df.drop(columns=[status_name, time_name])
    y = df[[status_name, time_name]]

    # normalization for features
    X = (X - X.min()) / (X.max() - X.min())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    input_dim, output_dim = len(X.columns), 1

    model = DeepWeiSurv(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

    model.eval()

    train_curve, valid_curve, shape_curve, scale_curve, ci_curve = [], [], [], [], []

    global_step = 0

    for t in range(EPOCHS):
        beta_ave, eta_ave = 0, 0
        torch.enable_grad()
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

        vloss, ci = validation(model, X_val, y_val)

        print("epoch:\t", t,
              "\t train loss:\t", "%.6f" % round(epoch_loss, 6),
              "\t valid loss:\t", "%.6f" % round(vloss, 6),
              "\t ave. shape:\t", "%.4f" % round(beta_ave * SHAPE_SCALE, 4),
              "\t ave. scale:\t", "%.3f" % round(eta_ave * SCALE_SCALE, 3),
              "\t concordance index:\t", "%.4f" % round(ci, 4))

        train_curve.append(epoch_loss), valid_curve.append(vloss)
        shape_curve.append(beta_ave * SHAPE_SCALE), scale_curve.append(eta_ave * SCALE_SCALE)
        ci_curve.append(ci)

    fig, axs = plt.subplots(5, figsize=[15, 20])
    axs[0].plot(train_curve), axs[0].set_title("Average Train Loss")
    axs[1].plot(valid_curve), axs[1].set_title("Average Validation Loss")
    axs[2].plot(shape_curve), axs[2].set_title("Average Shape Parameter")
    axs[3].plot(scale_curve), axs[3].set_title("Average Scale Parameter")
    axs[4].plot(ci_curve), axs[4].set_title("Concordance Index")
    fig.savefig("tests/img/dwsurv/dwsurv_{}_lr{}_{}.png".format(DATASET, LR, uuid.uuid1()))
    plt.show()
