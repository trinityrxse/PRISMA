'''
Training ANN to distinguish between VH, WWW and background

Not perfect:
Try and ensure final bin of the output nodes has only the desired data type
                (use the s_over_b function from output_nodes for this)
My data was FAST SIMULATION - hence I could not include the systematic variables
        - with not fast simulated data, include systematic variables in training

'''

import pandas as pd
# Make 3 classes
from keras.utils import to_categorical
# Scaling data
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from analysis_ import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import wandb
import joblib

def get_model(run):
    model = nn.Sequential(
        nn.Linear(23, run.config["hidden_layers"][0]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][0], run.config["hidden_layers"][1]),
        nn.Dropout(run.config["dropout"][0]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][1], run.config["hidden_layers"][2]),
        nn.Dropout(run.config["dropout"][1]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][2], run.config["hidden_layers"][3]),
        nn.Dropout(run.config["dropout"][2]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][3], run.config["hidden_layers"][4]),
        nn.Dropout(run.config["dropout"][3]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][4], run.config["hidden_layers"][5]),
        nn.Dropout(run.config["dropout"][4]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][5], run.config["hidden_layers"][6]),
        nn.Dropout(run.config["dropout"][5]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][6], run.config["hidden_layers"][7]),
        nn.Dropout(run.config["dropout"][6]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][7], run.config["hidden_layers"][8]),
        nn.Dropout(run.config["dropout"][7]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][8], run.config["hidden_layers"][9]),
        nn.Dropout(run.config["dropout"][8]),
        nn.SiLU(),
        nn.Linear(run.config["hidden_layers"][-1], 3),
        nn.Softmax(dim=1)
    )

    optimizer = optim.Adam(model.parameters(), lr=run.config["lr"])
    return model, optimizer

def train_MLP_multiclassifier(filepath):
        # load dataset
        df = pd.read_csv("/localscratch/df_preprocessed_.csv")

        x = df.drop('Type', axis=1)
        sc = RobustScaler()
        x = pd.DataFrame(sc.fit_transform(x))

        Y = df['Type'].values
        y_cat = to_categorical(Y, num_classes=3)

        class_weightvals = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)

        X = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
        y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)

        # categorical cross entropy, weighted
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([class_weightvals[0],
                                                        class_weightvals[1],
                                                        class_weightvals[2]]))
        n_epochs = 100

        # define configs to iterate through 
        config1 = {"hidden_layers": [126, 111, 108, 54, 27, 9, 3, 3, 3, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}

        config2 = {"hidden_layers": [126, 111, 108, 54, 27, 18, 9, 3, 3, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config3 = {"hidden_layers": [126, 111, 108, 75, 54, 27, 18, 9, 3, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config4 = {"hidden_layers": [102, 90, 75, 69, 54, 48, 36, 27, 18, 9],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "activation": 'silu',
                "lr": 0.01}
        config5 = {"hidden_layers": [90, 75, 69, 54, 48, 36, 27, 18, 9, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0],
                "activation": 'silu',
                "lr": 0.01}
        config6 = {"hidden_layers": [128, 116, 102, 80, 72, 64, 32, 24, 16, 8],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "activation": 'silu',
                "lr": 0.01}
        config7 = {"hidden_layers": [128, 112, 80, 48, 32, 16, 8, 3, 3, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config8 = {"hidden_layers": [128, 112, 80, 64, 48, 32, 16, 8, 3, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config9 = {"hidden_layers": [128, 112, 80, 64, 56, 48, 32, 16, 8, 3],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0],
                "activation": 'silu',
                "lr": 0.01}
        config10 = {"hidden_layers": [512, 264, 128, 104, 80, 64, 32, 24, 16, 8],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "activation": 'silu',
                "lr": 0.01}
        config11 = {"hidden_layers": [128, 116, 102, 80, 72, 64, 32, 24, 16, 8],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "activation": 'silu',
                "lr": 0.01}
        config12 = {"hidden_layers": [128, 116, 102, 80, 72, 64, 32, 24, 16, 8],
                "dropout": [0.37, 0.3, 0.26, 0.21, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config13 = {"hidden_layers": [512, 128, 116, 80, 72, 64, 32, 24, 16, 8],
                "dropout": [0.37, 0.3, 0.26, 0.21, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config14 = {"hidden_layers": [512, 128, 80, 64, 24, 8, 3, 3, 3, 3],
                "dropout": [0.37, 0.3, 0.26, 0.21, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config15 = {"hidden_layers": [1080, 512, 264, 128, 64, 32, 24, 16, 8, 4],
                "dropout": [0.37, 0.3, 0.26, 0.21, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config16 = {"hidden_layers": [1080, 512, 264, 128, 64, 32, 24, 16, 8, 3],
                "dropout": [0.37, 0.3, 0.26, 0.21, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}
        config17 = {"hidden_layers": [128, 116, 102, 80, 72, 64, 32, 24, 16, 8],
                "dropout": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "activation": 'silu',
                "lr": 0.01}
        config18 = {"hidden_layers": [1080, 512, 264, 128, 64, 32, 24, 16, 8, 4],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "activation": 'silu',
                "lr": 0.1}
        config19 = {"hidden_layers": [102, 90, 75, 69, 54, 48, 36, 27, 18, 9],
                "dropout": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "activation": 'silu',
                "lr": 0.1}
        config20 = {"hidden_layers": [102, 90, 75, 69, 54, 48, 36, 27, 18, 9],
                "dropout": [0.37, 0.3, 0.26, 0.21, 0.2, 0, 0, 0, 0],
                "activation": 'silu',
                "lr": 0.01}



        configs = [config1, config2, config3, config4, config5, config6, config7,
                   config8, config9, config10, config11, config12, config13,
                   config14, config15, config16, config17, config18, config19, config20]

        config_count=0
        accuracies = []
        for config in configs:
                config_count += 1
                print(config_count)
                run = wandb.init(project='mlp_torch', config=config)

                #for i in range(0, len(params)):
                model, optimizer = get_model(run)

                for epoch in range(n_epochs):
                        y_pred = model(X)
                        loss = loss_fn(y_pred, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        aucfunc = MultilabelAUROC(num_labels=3, thresholds=None)
                        auc = aucfunc(y_pred, y_int)
                        acc = (y_pred.round() == y).float().mean()

                        metrics = {
                        "train/train_loss": loss,
                        "train/epoch": epoch + 1,
                        "auc": auc,
                        "accuracy": acc,

                        }

                        wandb.log(metrics)

                accuracies.append(acc)
                wandb.finish()

                filename = f"{filepath}/MLP_torch_{config_count}.sav"
                joblib.dump(model, filename)

                y_pred = model(X)
                y_array = y.detach().numpy()
                y_pred = y_pred.detach().numpy()
                print(f'VH signal over background for config {config_count}: ', s_over_b(filepath, y_array, y_pred, '0', graph=False))
                print(f'WWW signal over background for config {config_count}: ', s_over_b(filepath, y_array, y_pred, '1', graph=False))
                plot_nodes_multiclassifier(filepath, y_pred, y_array)

        # find best accuracy model
        max_acc_idx = accuracies.index(max(accuracies))
        best_config = max_acc_idx + 1 # +1 since computers start at 0 
               
        print('End of Multiclassifier training')
        return best_config