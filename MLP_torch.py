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
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")

x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)

X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)
def get_model(layer_1=128, layer_2=80, layer_3=64, layer_4=32,
              layer_5=16, layer_6=8, layer_7=8, layer_8=8,
              layer_9=16, layer_10=8,
              dropout_1=0.37, dropout_2=0.3, dropout_3=0.26,
              dropout_4=0.2, dropout_5=0.17, dropout_6=0.15,
              dropout_7=0.12, dropout_8=0.1,
              lr=0.01):

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
        #nn.Dropout(dropout_5),
        #nn.SiLU(),
        #nn.Linear(layer_6, layer_7),
        #nn.Dropout(dropout_6),
        #nn.SiLU(),
        #nn.Linear(layer_7, layer_8),
        #nn.Dropout(dropout_7),
        #nn.SiLU(),
        #nn.Linear(layer_8, layer_9),
        #nn.Dropout(dropout_8),
        #nn.SiLU(),
        #nn.Linear(layer_9, layer_10),
        nn.SiLU(),
        nn.Linear(layer_6, 3),
        nn.Softmax(dim=1)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


params = {"layer_1": [np.arange(16, 128, 8, dtype=int)],
          "layer_2": [np.arange(16, 128, 8, dtype=int)],
          "layer_3": [np.arange(16, 128, 8, dtype=int)],
          "layer_4": [np.arange(16, 128, 8, dtype=int)],
          "dropout_1": [np.arange(0.2, 0.4, 0.05, dtype=float)],
          "dropout_2": [np.arange(0.2, 0.4, 0.05, dtype=float)],
          "dropout_3": [np.arange(0.2, 0.4, 0.05, dtype=float)],
          "lr": [1., 0.1, 0.01, 0.001]}

# categorical cross entropy, weighted
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([class_weight[0],
                                                   class_weight[1],
                                                   class_weight[2]]))
n_epochs = 300


#for i in range(0, len(params)):
model, optimizer = get_model()
wandb.init(project='mlp_torch')
for epoch in range(n_epochs):
    Xbatch = X
    y_pred = model(Xbatch)
    ybatch = y
    loss = loss_fn(y_pred, ybatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    aucfunc = MultilabelAUROC(num_labels=3, thresholds=None)
    auc = aucfunc(y_pred, y_int)
    acc = (y_pred.round() == ybatch).float().mean()
    y_array = y.detach().numpy()
    y_pred = y_pred.detach().numpy()
    VH_sig_over_bkg = sigoverbkg('VH', y_pred, y_array)
    WWW_sig_over_bkg = sigoverbkg('WWW', y_pred, y_array)

    metrics = {
        "train/train_loss": loss,
        "train/epoch": epoch + 1,
        "auc": auc,
        "accuracy": acc,
        "VH_sig_over_bkg": VH_sig_over_bkg,
        "WWW_sig_over_bkg": WWW_sig_over_bkg,
    }
    wandb.log(metrics)

plot_nodes(y_array, y_pred)
VH_sig_over_bkg = sigoverbkg('VH', y_pred, y_array, graph=True)
WWW_sig_over_bkg = sigoverbkg('WWW', y_pred, y_array, graph=True)

filename = "/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_torch_VH.sav"
joblib.dump(model, filename)
