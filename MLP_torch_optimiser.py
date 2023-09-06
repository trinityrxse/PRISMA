import pandas as pd
# Make 3 classes
from keras.utils import to_categorical
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from analysis import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import wandb
import joblib

# load dataset
df = pd.read_csv("/localscratch/df_preprocessed_.csv")

x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)

X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)
def get_model(run):
    layer_1 = run.config["hidden_layers"][0]
    layer_2 = run.config["hidden_layers"][1]
    layer_3 = run.config["hidden_layers"][2]
    layer_4 = run.config["hidden_layers"][3]
    layer_5 = run.config["hidden_layers"][4]
    layer_6 = run.config["hidden_layers"][5]
    layer_7 = run.config["hidden_layers"][6]
    layer_8 = run.config["hidden_layers"][7]
    layer_9 = run.config["hidden_layers"][8]
    layer_10 = run.config["hidden_layers"][9]
    dropout_1 = run.config["dropout"][0]
    dropout_2 = run.config["dropout"][1]
    dropout_3 = run.config["dropout"][2]
    dropout_4 = run.config["dropout"][3]
    dropout_5 = run.config["dropout"][4]
    dropout_6 = run.config["dropout"][5]
    dropout_7 = run.config["dropout"][6]
    dropout_8 = run.config["dropout"][7]
    dropout_9 = run.config["dropout"][8]
    lr = run.config["lr"]

    model = nn.Sequential(
        nn.Linear(27, layer_1),
        nn.SiLU(),
        nn.Linear(layer_1, layer_2),
        nn.Dropout(dropout_1),
        nn.SiLU(),
        nn.Linear(layer_2, layer_3),
        nn.Dropout(dropout_2),
        nn.SiLU(),
        nn.Linear(layer_3, layer_4),
        nn.Dropout(dropout_3),
        nn.SiLU(),
        nn.Linear(layer_4, layer_5),
        nn.Dropout(dropout_4),
        nn.SiLU(),
        nn.Linear(layer_5, layer_6),
        nn.Dropout(dropout_5),
        nn.SiLU(),
        nn.Linear(layer_6, layer_7),
        nn.Dropout(dropout_6),
        nn.SiLU(),
        nn.Linear(layer_7, layer_8),
        nn.Dropout(dropout_7),
        nn.SiLU(),
        nn.Linear(layer_8, layer_9),
        nn.Dropout(dropout_8),
        nn.SiLU(),
        nn.Linear(layer_9, layer_10),
        nn.Dropout(dropout_9),
        nn.SiLU(),
        nn.Linear(layer_10, 3),
        nn.Softmax(dim=1)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

# categorical cross entropy, weighted
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([class_weight[0],
                                                   class_weight[1],
                                                   class_weight[2]]))
n_epochs = 100

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



configs = [config10, config11, config12,
           config13, config14, config15, config16, config17, config18, config19, config20]

for config in configs:

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
        #y_array = y.detach().numpy()
        #y_pred = y_pred.detach().numpy()
        #VH_sig_over_bkg = s_over_b(y_pred, y_array, '0')
        #WWW_sig_over_bkg = s_over_b(y_pred, y_array, '1')

        metrics = {
            "train/train_loss": loss,
            "train/epoch": epoch + 1,
            "auc": auc,
            "accuracy": acc,
            #"VH_sig_over_bkg": VH_sig_over_bkg,
            #"WWW_sig_over_bkg": WWW_sig_over_bkg,
            }

        wandb.log(metrics)

    filename = "/localscratch/MLP_torch.sav"
    joblib.dump(model, filename)

    y_pred = model(X)
    y_array = y.detach().numpy()
    y_pred = y_pred.detach().numpy()
    plot_nodes_re(y_pred, y_array)
    #VH_sig_over_bkg = s_over_b(y_array, y_pred, '0')
    #WWW_sig_over_bkg = s_over_b(y_array, y_pred, '1')

    wandb.finish()

