import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import numba as nb
"""""
from keras.layers import Dense, Dropout
from keras.models import Sequential
# Make 3 classes
from keras.utils import to_categorical
"""""
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from analysis import *

# load dataset
df_SM = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")
df_EFT = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_cHq3_-1.csv")

df_SM.insert(loc=0, column="SM", value=1)
df_EFT.insert(loc=0, column="SM", value=0)

df = pd.concat([df_SM, df_EFT], axis=0, ignore_index=True)

cut = (df["Type"] == 0) | (df["Type"] == 1)
df = df[cut]

df = df.drop("Type", axis=1)
df = df.dropna(axis=0)
x = df.drop("SM", axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['SM'].values

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.1, shuffle=True)

class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC
X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

def EFT_model(run):
    if run.config["activation"] == 'relu':
        print(run.config["hidden_layers"][-1])
        model = nn.Sequential(
            nn.Linear(23, run.config["hidden_layers"][0]),
            nn.ReLU(),
            nn.Linear(run.config["hidden_layers"][0], run.config["hidden_layers"][1]),
            nn.Dropout(run.config["dropout"][0]),
            nn.ReLU(),
            nn.Linear(run.config["hidden_layers"][1], run.config["hidden_layers"][2]),
            nn.Dropout(run.config["dropout"][1]),
            nn.ReLU(),
            nn.Linear(run.config["hidden_layers"][2], run.config["hidden_layers"][3]),
            nn.Dropout(run.config["dropout"][2]),
            #nn.ReLU(),
            #nn.Linear(run.config["hidden_layers"][3], run.config["hidden_layers"][4]),
            #nn.Dropout(run.config["dropout"][3]),
            #nn.ReLU(),
            #nn.Linear(run.config["hidden_layers"][4], run.config["hidden_layers"][5]),
            #nn.Dropout(run.config["dropout"][4]),
            #nn.ReLU(),
            #nn.Linear(run.config["hidden_layers"][5], run.config["hidden_layers"][6]),
            #nn.Dropout(run.config["dropout"][5]),
            #nn.ReLU(),
            #nn.Linear(run.config["hidden_layers"][6], run.config["hidden_layers"][7]),
            #nn.Dropout(run.config["dropout"][6]),
            nn.ReLU(),
            nn.Linear(run.config["hidden_layers"][-1], 1),
            nn.Sigmoid()
        )
    elif run.config["activation"] == 'silu':
        print(run.config["hidden_layers"][-1])
        model = nn.Sequential(
            nn.Linear(23, run.config["hidden_layers"][0]),
            nn.SiLU(),
            nn.Linear(run.config["hidden_layers"][0], run.config["hidden_layers"][1]),
            nn.Dropout(run.config["dropout"][0]),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.Dropout(run.config["dropout"][1]),
            nn.SiLU(),
            nn.Linear(run.config["hidden_layers"][-1], 1),
            nn.Sigmoid()
        )

    optimizer = optim.Adam(model.parameters(), lr=run.config["lr"])

    return model, optimizer

config = {"hidden_layers": [128, 64, 32, 16],
          "dropout": [0, 0, 0, 0, 0, 0, 0],
          "activation": 'relu',
          "lr": 0.001}

run = wandb.init(project='mlp_eft', config=config)

n_epochs = 100
loss_fn = nn.BCELoss()  # binary cross entropy
model, optimizer = EFT_model(run)
for epoch in range(n_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    B_AUROC = BinaryAUROC(thresholds=None)
    auc = B_AUROC(y_pred, y)
    acc = (y_pred.round() == y).float().mean()
    #y_array = y.detach().numpy()
    #y_array = nb.typed.List([i[0] for i in y_array])
    #y_pred = y_pred.detach().numpy()
    #y_pred = nb.typed.List([i[0] for i in y_pred])
    #EFT_truth = EFT_s_over_b('EFT', y_array, y_pred, get_weights_EFT('cHq3_-1', '0[0-4]'))
    #SM_truth = EFT_s_over_b('SM', y_array, y_pred, get_weights_EFT('cHq3_-1', '0[0-4]'))

    metrics = {
        "train/train_loss": loss,
        "train/epoch": epoch + 1,
        "auc": auc,
        "accuracy": acc,
        #"EFT over SM at 0": EFT_truth,
        #"SM over EFT at 1": SM_truth,
    }
    wandb.log(metrics)

y_pred = model(X)
y_array = y.detach().numpy()
y_array = nb.typed.List([i[0] for i in y_array])
y_pred = y_pred.detach().numpy()
y_pred = nb.typed.List([i[0] for i in y_pred])
plot_nodes_2(y_array, y_pred, get_weights_EFT('cHq3_-1', '0[0-4]'))

filename = "/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_EFT.sav"
joblib.dump(model, filename)