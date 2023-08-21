import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.utils import to_categorical

filename = '/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/initial_model.sav'

# load the model from disk
model = joblib.load(filename)

# load dataset
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed.csv")
df.shape
df.keys()

x = df.drop('Type', axis=1)
x.shape
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2)


pred_train = model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model.predict(x_test)

scores2 = model.evaluate(x_test, y_test, verbose=0)

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

y_pred = model.predict(x_test)

y_pred_class = np.argmax(y_pred, axis=1)

y_test_class = np.argmax(y_test, axis=1)

featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met', 'delR_l0l1', 'delR_l0l2',
                'delR_l1l2', 'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2', 'dPhi_MET_l0',
                'dPhi_MET_l1', 'dPhi_MET_l2', 'max_PT_jet', 'd0_l0', 'd0_l1', 'd0_l2',
                'z0sintheta_l0', 'z0sintheta_l1', 'z0sintheta_l2', 'n_btag', 'mT_l0l1',
                'mT_l0l2', 'mT_l1l2', 'sumPT', 'm_l0l1', 'm_l0l2', 'm_l1l2', 'm_lll', 'F_alpha']

import glob

lhe_signal = glob.glob(
    "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].lhe.gz")
root_signal = glob.glob(
    "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].root")

lhe_bkg = glob.glob("/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.lhe.gz")
root_bkg = glob.glob("/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.root")

N_lhe_sig = 1000000  # getNeventsLHE(lhe_signal)
N_lhe_bkg = 4000000  # getNeventsLHE(lhe_bkg)

N_root_sig = 511723  # getNeventsRoot(root_signal)
N_root_bgk = 2964674  # getNeventsRoot(root_bkg)

xSection_sig = get_xSection(lhe_signal)
xSection_bkg = get_xSection(lhe_bkg)

# calculate resulting xSection*efficiency
xSection_sig *= N_root_sig / N_lhe_sig
xSection_bkg *= N_root_bgk / N_lhe_bkg

# scale to given luminosity
lumi = 400  # inverse fb

n_sig = lumi * xSection_sig * 1000
n_bkg = lumi * xSection_bkg * 1000

weight_sig = n_sig / N_root_sig
weight_bkg = n_bkg / N_root_bgk

x_test = np.array(x_test)
df_test = pd.DataFrame(x_test, columns=featurenames)
df_test.insert(loc=0, column='Type', value=y_test_class)

df_VH_test = df_test[df_test['Type'] == 0]
df_WWW_test = df_test[df_test['Type'] == 1]
df_bkg_test = df_test[df_test['Type'] == 2]

plothisttriple(df_VH_test, df_WWW_test, df_bkg_test, ["PT_l0", "PT_l1", "PT_l2"], 400., 800., 1500., 10, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH_test, df_WWW_test, df_bkg_test, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 1500., 2000., 20, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH_test, df_WWW_test, df_bkg_test, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 2.e5, 6.e5, 8.e5, 16, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH_test, df_WWW_test, df_bkg_test, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4.1, 0.9, 1.3, 8, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH_test, df_WWW_test, df_bkg_test, ["d0_l0", "d0_l1", "d0_l2"], 1., float(np.max(df["d0_l1"])), float(np.max(df["d0_l2"])), 10, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH_test, df_WWW_test, df_bkg_test, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., float(np.max(df["delR_l0l2"])), float(np.max(df["delR_l1l2"])), 30, weight_sig, weight_bkg, lumi)
plothist(df_VH_test, df_WWW_test, df_bkg_test, "m_lll", 0., 1500., 20, weight_sig, weight_bkg, lumi)
plothist(df_VH_test, df_WWW_test, df_bkg_test, "sumPT", 0., 1500., 100, weight_sig, weight_bkg, lumi)
plothist(df_VH_test, df_WWW_test, df_bkg_test, "n_btag", 0., 10., 9, weight_sig, weight_bkg, lumi)
plothist(df_VH_test, df_WWW_test, df_bkg_test, "max_PT_jet", 20., 200., 20, weight_sig, weight_bkg,lumi)
plothist(df_VH_test, df_WWW_test, df_bkg_test, "met", 0., 500., 100, weight_sig, weight_bkg, lumi)
plothist(df_VH_test, df_WWW_test, df_bkg_test, "F_alpha", 0., .0065, 20, weight_sig, weight_bkg, lumi)


# confusion_matrix(y_test_class, y_pred_class)

print(classification_report(y_test_class, y_pred_class))
