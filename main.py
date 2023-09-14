#Analysis Packages
import numpy as np
import pandas as pd
from analysis import *

#Plotting
import matplotlib.pyplot as plt

#Data Formatting
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Classification Report
from sklearn.metrics import classification_report

#Load Cross Validation
#from ML_Kfold import KFold_CV
#Load Model
import joblib
import torch

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

# df of all data provided
# can read in dfs from EFTs
df = pd.read_csv("/localscratch/df_preprocessed_.csv")

"""""
#plot histograms for all data
plothist(df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50)
plothist(df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50)
plothist(df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30)
plothist(df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30)
plothist(df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30)
plothist(df, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4., 30)
plothist(df, ["m_lll"], 1500., 50)
plothist(df, ["sumPT"], 1500., 100)
plothist(df, ["n_btag"], 10., 9)
plothist(df, ["met"], 500., 100)
plothist(df, ["max_PT_jet"], 500., 100)
plothist(df, ["F_alpha"], 1., 20)
"""""

# load trained ML model optimised for both VH and WWW separation
filename = f"/localscratch/MLP_torch_{config12}.sav"
#filename = '/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/initial_model.sav'
model = joblib.load(filename)

# format data to use model on it
x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2)

X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)

print(X.shape, 'works')
# Get df post-training using the classifications predicted by the model 'y_test_class'
y_pred = model(X)
y_array = y.detach().numpy()
y_pred = y_pred.detach().numpy()

df_test = get_df_nodes(y_pred, y_array)
s_over_b(y_array, y_pred, '0', graph=True)
s_over_b(y_array, y_pred, '1', graph=True)

plot_nodes_re(y_pred, y_array)

y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using model


print(classification_report(y_truth_class, y_pred_class))

featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met',
                   'delR_l0l1', 'delR_l0l2', 'delR_l1l2',
                   'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2',
                   'dPhi_MET_l0', 'dPhi_MET_l1', 'dPhi_MET_l2','max_PT_jet',
                   'mT_l0l1', 'mT_l0l2', 'mT_l1l2', 'sumPT',
                   'm_l0l1', 'm_l0l2', 'm_l1l2',
                   'm_lll', 'F_alpha']

x = np.array(sc.inverse_transform(x))
df_test = pd.DataFrame(x, columns=featurenames)
df_test.insert(loc=0, column='Type', value=y_pred_class)

# TODO fix K fold CV for pytorch network
# make df of post-training data
#df_test = KFold_CV(df, model)

weight_sig, weight_bkg, lumi = get_weights()
#histograms to compare input and output hists
"""""
plothist(df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["m_lll"], 1500., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["sumPT"], 1500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["met"], 500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["max_PT_jet"], 500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["F_alpha"], 1., 20, df2=df_test, weight2=weight_sig, name2='Post-ML')
"""""
# cut on bkg output
# make the WWW-VH data 
df_test = df_test.drop('Type', axis=1)

df_test.insert(loc=0, column='Pred Type', value=y_pred_class)
df_test.insert(loc=0, column='Type', value=df['Type'].values)
df_test_wo = df_test[df_test["Pred Type"] != 2] #remove everything predicted as bkg

# format data to use model on it
x = df_test_wo.drop('Type', axis=1)
x = x.drop('Pred Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df_test_wo['Type'].values
y_cat = to_categorical(Y, num_classes=3)

X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)

# Get df post-training using the classifications predicted by the model 'y_test_class'
y_pred = model(X)
y_array = y.detach().numpy()
y_pred = y_pred.detach().numpy()

array_list = plot_nodes_re(y_pred, y_array, ret=True)
array_VH = array_list[0]
array_WWW = array_list[1]
array_bkg = array_list[2]

y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using model

print(classification_report(y_truth_class, y_pred_class))

x = np.array(sc.inverse_transform(x))
df_test2 = pd.DataFrame(x, columns=featurenames)
df_test2.insert(loc=0, column='Type', value=y_pred_class)

"""""
plothist(df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["m_lll"], 1500., 50, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["sumPT"], 1500., 100, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["met"], 500., 100, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["max_PT_jet"], 500., 100, df2=df_test2, weight2=weight_sig, name2='Post-ML')
plothist(df, ["F_alpha"], 1., 20, df2=df_test2, weight2=weight_sig, name2='Post-ML')
"""""

def make_delta_node(array):
    delta_node = []
    for i in range(0, len(array[0])):
        delta_node.append(array[1][i] - array[0][i])

    return delta_node

delta_node = make_delta_node(array_VH)
print(delta_node)
print(array_VH[2])

import pyhf
from pyhf.contrib.viz import brazil

bkg_uncertainty = list(np.sqrt(array_VH[2] * weight_bkg))
print(bkg_uncertainty)


pyhf.set_backend("numpy")
model = pyhf.simplemodels.uncorrelated_background(signal=(delta_node + delta_node), 
                                                  bkg=(list(array_VH[2]) + list(array_VH[2])), 
                                                  bkg_uncertainty=(bkg_uncertainty + bkg_uncertainty))

poi_vals = np.linspace(0, 0.5, 41)
data = (list(array_VH[2]) + list(array_VH[2])) + model.config.auxdata

best_fit_pars = pyhf.infer.mle.fit(data, model)
print(f"initialization parameters: {model.config.suggested_init()}")
print(
    f"best fit parameters:\
    \n * signal strength: {best_fit_pars[0]}\
    \n * nuisance parameters: {best_fit_pars[1:]}"
)

results = [
    pyhf.infer.hypotest(test_poi, data, model, test_stat='qtilde', return_expected_set=True) for test_poi in poi_vals
]

fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
brazil.plot_results(poi_vals, results, ax=ax)
plt.show()

print('WHERE IS MY GRAPH')