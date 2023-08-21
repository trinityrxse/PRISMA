# Analysis Packages
import numpy as np
import pandas as pd
from analysis import *

# Plotting
import matplotlib.pyplot as plt

# Data Formatting
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Classification Report
from sklearn.metrics import classification_report

# Load Model
import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
import keras
from sklearn.utils import class_weight

#TODO get weights to work for the different samples
weight_sig, weight_bkg, lumi = get_weights() #weights & luminosity

# df of all data provided
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed.csv")

x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.1)

class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
class_weight = {0: class_weight[0],
                1: class_weight[1],
                2: class_weight[2]}
callback_model = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    verbose=0,
    mode="auto",
)
for j in range(0, 5):
    model_VH = joblib.load(f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/VH_tuned_model_{j}.sav')
    model_WWW = joblib.load(f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/WWW_tuned_model_{j}.sav')
    if model_VH.summary() == model_WWW.summary():
        model_both = model_VH
        model_history = model_both.fit(x_train, y_train, verbose=1, epochs=100, \
                                     class_weight=class_weight, callbacks=[callback_model],
                                     validation_data=(x_test, y_test))

        y_pred = model_both.predict(x_test)
        print(max(y_pred[0]))
        sigoverbkg_VH(y_pred, y_test, name=j)
        sigoverbkg_WWW(y_pred, y_test, name=j)
        plot_nodes(node_outputs(y_test, y_pred))
        filename = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/both_tuned_model_{j}.sav'
        joblib.dump(model_both, filename)


    else:
        model_history_VH = model_VH.fit(x_train, y_train, verbose=1, epochs=100, \
                                       class_weight=class_weight, callbacks=[callback_model],
                                       validation_data=(x_test, y_test))
        model_history_WWW = model_WWW.fit(x_train, y_train, verbose=1, epochs=100, \
                                       class_weight=class_weight, callbacks=[callback_model],
                                       validation_data=(x_test, y_test))

        y_pred_VH = model_VH.predict(x_test)
        y_pred_WWW = model_WWW.predict(x_test)
        print(max(y_pred_VH[0]))
        sigoverbkg_VH(y_pred_VH, y_test)
        sigoverbkg_WWW(y_pred_WWW, y_test)
        plot_nodes(node_outputs(y_test, y_pred_VH))
        plot_nodes(node_outputs(y_test, y_pred_WWW))

        filename_VH = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/VH_tuned_model_{j}.sav'
        filename_WWW = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/WWW_tuned_model_{j}.sav'
        joblib.dump(model_VH, filename_VH)
        joblib.dump(model_WWW, filename_WWW)


