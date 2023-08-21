import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
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

"""""
class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
print(class_weight)
class_weight = {0: class_weight[0],
                1: class_weight[1]}
callback = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    verbose=0,
    mode="auto",
)

def get_model_EFT(init_mode='lecun_uniform', optimizer='adam'):
    model = Sequential()
    model.add(Dense(34, kernel_initializer=init_mode, input_shape=(27,), activation='relu'))
    model.add(Dense(34, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='softmax'))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['AUC'])
    model.summary()
    return model


model = get_model_EFT()
model_history = model.fit(x_train, y_train, verbose=1, epochs=100, \
                          class_weight=class_weight, callbacks=[callback], validation_data=(x_test, y_test))


# Random Forest
from sklearn.ensemble import RandomForestClassifier
# create model instance
rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=90,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
# fit model
rf.fit(x_train,y_train)
# make predictions
y_pred_rf = rf.predict(x_test)

# XGBoost GBM
from xgboost import XGBClassifier
# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst.fit(x_train, y_train)
# make predictions
y_pred_bst = bst.predict(x_test)

# Define estimators
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimator_list = [
    ('dt',dt),
    ('rf',rf),
    ('bst',bst) ]

# Build stack model
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression()
)

# Train stacked model
stack_model.fit(x_train, y_train)
"""""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC
X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)


model = nn.Sequential(
    nn.Linear(27, 64),
    nn.ReLU(),
    #nn.Linear(64, 32),
    #nn.Dropout(0.2),
    #nn.ReLU(),
    #nn.Linear(32, 16),
    #nn.Dropout(0.2),
    #nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

n_epochs = 100
batch_size = len(X)

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

wandb.init(project='mlp_eft')
for epoch in range(n_epochs):
    Xbatch = X
    y_pred = model(Xbatch)
    ybatch = y
    loss = loss_fn(y_pred, ybatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    B_AUROC = BinaryAUROC(thresholds=None)
    auc = B_AUROC(y_pred, ybatch)
    acc = (y_pred.round() == ybatch).float().mean()
    y_array = y.detach().numpy()
    y_array = [i[0] for i in y_array]
    y_pred = y_pred.detach().numpy()
    y_pred = [i[0] for i in y_pred]
    EFT_truth = EFT_s_over_b(y_array, y_pred, weight=get_weights_EFT('cHq3_-1', '0[0-4]'))

    metrics = {
        "train/train_loss": loss,
        "train/epoch": epoch + 1,
        "auc": auc,
        "accuracy": acc,
        "EFT_truth": EFT_truth,
    }
    wandb.log(metrics)

plot_nodes_2(y, y_pred, weight=get_weights_EFT('cHq3_-1', '0[0-4]'))

filename = "/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_EFT.sav"
joblib.dump(model, filename)