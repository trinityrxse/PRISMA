import numpy as np
import pandas as pd
from analysis import *
import numba as nb
# Plotting
import matplotlib.pyplot as plt

# Data Formatting
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Classification Report
from sklearn.metrics import classification_report

# Load Cross Validation
from ML_Kfold import KFold_CV
# Load Model
import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
import keras
import numba
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
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC

weight_sig, weight_bkg, lumi = get_weights()  # weights & luminosity

# df of all SM data
df_SM = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")

# load trained ML model optimised for both VH and WWW separation
filename = "/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_torch_new.sav"
model = joblib.load(filename)

# Use model on SM data
# format data to use model on it
x = df_SM.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df_SM['Type'].values

y_cat = to_categorical(Y, num_classes=3)

class_weight_SM = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)

X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)

# Get df post-training using the classifications predicted by the model 'y_test_class'
y_pred = model(X)
y_array = y.detach().numpy()
y_pred = y_pred.detach().numpy()
VH_sig_over_bkg = sigoverbkg('VH', y_pred, y_array)
WWW_sig_over_bkg = sigoverbkg('WWW', y_pred, y_array)

y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using model

print(classification_report(y_truth_class, y_pred_class))

featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met',
                'delR_l0l1', 'delR_l0l2', 'delR_l1l2',
                'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2',
                'dPhi_MET_l0', 'dPhi_MET_l1', 'dPhi_MET_l2',
                'z0sintheta_l0', 'z0sintheta_l1', 'z0sintheta_l2',
                'n_btag', 'max_PT_jet',
                'mT_l0l1', 'mT_l0l2', 'mT_l1l2', 'sumPT',
                'm_l0l1', 'm_l0l2', 'm_l1l2',
                'm_lll', 'F_alpha']

x = np.array(sc.inverse_transform(x))
df_test = pd.DataFrame(x, columns=featurenames)
df_test.insert(loc=0, column='Type', value=y_pred_class)


#data_list = ["cHq1_-1", "cHq1_1", "cHq3_-1", "cHq3_1", "cHW_-1",
#             "cHW_1", "cW_-1", "cW_1"]
data_list = ["cHq3_-1"]
#number_events = ["0[0-4]", "0[0-4]", "0[0-4]", "0[0-9]", "[0-1]?", "[0-1]?", "[0-1]?", "[0-1]?"]
number_events = ["0[0-4]"]

#training the MLP for EFT vs SM distinction

for i in range(0, len(data_list)):

    #plots histograms for EFT inputs before and after classification by model
    #returns df of EFT and SM data with predicted classes (truth class not a variable anymore)

    df_EFT_NN = get_VH_WWW_df(model, data_list[i], number_events[i],
                              df_test, y_pred, y_array)
    print(df_EFT_NN.shape)

    weightEFT = get_weights_EFT(data_list[i], number_events[i])

    df_EFT_NN = df_EFT_NN.drop("Type", axis=1)
    df_EFT_NN = df_EFT_NN.dropna(axis=0)
    x = df_EFT_NN.drop("SM", axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))

    Y = df_EFT_NN['SM'].values

    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)

    filenameEFT = f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_EFT.sav"
    model_EFT = joblib.load(filenameEFT)

    y_pred = model_EFT(X)
    y_array = y.detach().numpy()
    y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using mode
    y_array = nb.typed.List([i[0] for i in y_array])
    y_pred = y_pred.detach().numpy()
    y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
    y_pred = nb.typed.List([i[0] for i in y_pred])

    plot_nodes_2(y_array, y_pred, get_weights_EFT(data_list[i], number_events[i]))
    print(classification_report(y_truth_class, y_pred_class))




