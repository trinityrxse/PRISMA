import joblib
import numpy as np
import pandas as pd
import wandb
import numba as nb

# Model
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC

# Scaling data
from sklearn.preprocessing import RobustScaler
from analysis import *


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

def train_EFT_model(filepath):
    # load dataset
    df_SM = pd.read_csv(f"{filepath}/df_preprocessed_.csv")
    df_EFT = pd.read_csv(f"{filepath}/df_preprocessed_cHq3_-1.csv")

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

    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

        #choose your own configs
    config1 = {"hidden_layers": [128, 64, 32, 16],
            "dropout": [0, 0, 0],
            "activation": 'relu',
            "lr": 0.01}

    config2 = {"hidden_layers": [128, 64, 32, 8],
            "dropout": [0, 0, 0],
            "activation": 'relu',
            "lr": 0.01}

    config3 = {"hidden_layers": [264, 128, 64, 32],
            "dropout": [0, 0, 0],
            "activation": 'relu',
            "lr": 0.01}

    config4 = {"hidden_layers": [64, 32, 16, 8],
            "dropout": [0, 0, 0],
            "activation": 'relu',
            "lr": 0.01}

    config5 = {"hidden_layers": [128, 64, 32, 16],
            "dropout": [0.2, 0.2, 0.2],
            "activation": 'relu',
            "lr": 0.01}

    config6 = {"hidden_layers": [128, 64, 32, 8],
            "dropout": [0.2, 0.2, 0.2],
            "activation": 'relu',
            "lr": 0.01}

    config7 = {"hidden_layers": [264, 128, 64, 32],
            "dropout": [0.2, 0.2, 0.2],
            "activation": 'relu',
            "lr": 0.01}

    config8 = {"hidden_layers": [64, 32, 16, 8],
            "dropout": [0.2, 0.2, 0.2],
            "activation": 'relu',
            "lr": 0.01}

    config9 = {"hidden_layers": [128, 64, 32, 16],
            "dropout": [0, 0, 0],
            "activation": 'relu',
            "lr": 0.001}

    config10 = {"hidden_layers": [64, 32, 16, 8],
            "dropout": [0, 0, 0],
            "activation": 'relu',
            "lr": 0.001}

    configs = [config1, config2, config3, config4, config5, config6,
            config7, config8, config9, config10]
    config_counter = 0
    for config in configs:
        config_counter += 1
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
            metrics = {
                "train/train_loss": loss,
                "train/epoch": epoch + 1,
                "auc": auc,
                "accuracy": acc,
            }
            wandb.log(metrics)

        wandb.finish()

        y_pred = model(X)
        y_array = y.detach().numpy()
        y_array = nb.typed.List([i[0] for i in y_array])
        y_pred = y_pred.detach().numpy()
        y_pred = nb.typed.List([i[0] for i in y_pred])

        filename = f"{filepath}/MLP_EFT_{config_counter}.sav"
        joblib.dump(model, filename)

        plot_nodes_2(filepath, y_array, y_pred, get_weights_EFT(filepath, 'cHq3_-1'))
