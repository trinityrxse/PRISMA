import joblib
import keras
import numpy as np
import pandas as pd
# Make 3 classes
from keras.utils import to_categorical
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from analysis import *
#Classification Report
from sklearn.metrics import classification_report
import wandb
from wandb.keras import WandbMetricsLogger

# load dataset
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")

x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2)

class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
class_weight = {0: class_weight[0],
                1: class_weight[1],
                2: class_weight[2]}
callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=0,
    mode="auto",
)



VH_over_bkg = []
WWW_over_bkg = []
for j in range(3,4):
    filename_VH = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/VH_tuned_model_{j}.sav'
    filename_WWW = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/WWW_tuned_model_{j}.sav'
    model_VH = joblib.load(filename_VH)
    model_WWW = joblib.load(filename_WWW)
    #print(model_VH.summary())
    #print(model_WWW.summary())

    if model_VH.summary() == model_WWW.summary():
        wandb.init()
        model_both = model_VH
        model_history = model_both.fit(x_train, y_train, verbose=1, epochs=100, \
                                  class_weight=class_weight, callbacks=[callback, WandbMetricsLogger()], validation_data=(x_test, y_test))

        plot_modelhistory(model_history, 'loss')
        plot_modelhistory(model_history, 'auc')

        y_pred_test = model_both.predict(x_test)
        y_pred_train = model_both.predict(x_train)
        y_pred = np.concatenate((y_pred_train, y_pred_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        VH_over_bkg.append([j, sigoverbkg('VH', y_pred, y, name_trial=j)])
        WWW_over_bkg.append([j, sigoverbkg('WWW', y_pred, y, name_trial=j)])

        y_pred_class = np.argmax(y_pred_test, axis=1)  # classes predicted for training data
        y_test_class = np.argmax(y_test, axis=1)  # classes predicted for test data using model

        print(classification_report(y_test_class, y_pred_class))

        filename_both = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/both_tuned_model_{j}_100.sav'
        joblib.dump(model_both, filename_both)
        wandb.finish()
    else:
        wandb.init()
        model_history_VH = model_VH.fit(x_train, y_train, verbose=1, epochs=100, \
                                  class_weight=class_weight, callbacks=[callback], validation_data=(x_test, y_test))
        wandb.finish()
        wandb.init()
        model_history_WWW = model_WWW.fit(x_train, y_train, verbose=1, epochs=100, \
                                  class_weight=class_weight, callbacks=[callback], validation_data=(x_test, y_test))
        wandb.finish()
        plot_modelhistory(model_history_VH, 'loss')
        plot_modelhistory(model_history_VH, 'auc')
        plot_modelhistory(model_history_WWW, 'loss')
        plot_modelhistory(model_history_WWW, 'auc')

        #needed for both model evals
        y = np.concatenate((y_train, y_test), axis=0)
        y_test_class = np.argmax(y_test, axis=1)  # classes predicted for test data using model

        #for VH model
        y_pred_test_VH = model_VH.predict(x_test)
        y_pred_train_VH = model_VH.predict(x_train)
        y_pred_VH = np.concatenate((y_pred_train_VH, y_pred_test_VH), axis=0)

        # append scores but note which model it was from
        VH_over_bkg.append([f"{j} VH", sigoverbkg('VH', y_pred_VH, y, name_trial=f"{j} VH")])
        WWW_over_bkg.append([f"{j} VH", sigoverbkg('WWW', y_pred_VH, y, name_trial=f"{j} VH")])

        y_pred_class_VH = np.argmax(y_pred_test_VH, axis=1)  # classes predicted for training data
        print(classification_report(y_test_class, y_pred_class_VH))

        #for WWW model
        y_pred_test_WWW = model_VH.predict(x_test)
        y_pred_train_WWW = model_VH.predict(x_train)
        y_pred_WWW = np.concatenate((y_pred_train_WWW, y_pred_test_WWW), axis=0)

        #append scores but note which model it was from
        VH_over_bkg.append([f"{j} WWW", sigoverbkg('VH', y_pred_WWW, y, name_trial=f"{j} WWW")])
        WWW_over_bkg.append([f"{j} WWW", sigoverbkg('WWW', y_pred_WWW, y, name_trial=f"{j} WWW")])

        y_pred_class_WWW = np.argmax(y_pred_test_WWW, axis=1)  # classes predicted for training data
        print(classification_report(y_test_class, y_pred_class_WWW))

        filename_VH = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/VH_tuned_model_{j}_100.sav'
        joblib.dump(model_VH, filename_VH)
        filename_VH = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/WWW_tuned_model_{j}_100.sav'
        joblib.dump(model_WWW, filename_WWW)

