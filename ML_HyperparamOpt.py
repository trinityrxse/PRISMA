import joblib
import keras
import keras_tuner as kt
import pandas as pd
from keras.models import Sequential
# Classification Report
from keras.utils import to_categorical
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from keras.layers import Dense, Dropout
from time import time
from analysis import *
import numpy as np


class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse')
    return model


# Optimising Hyperparameters
def model_builder(hp):
    model = Sequential()
    hp_activation = hp.Choice('activation', values=['swish', 'relu', 'sigmoid'])
    hp_layer_1 = hp.Int('layer_1', min_value=16, max_value=128, step=16)
    hp_layer_2 = hp.Int('layer_2', min_value=16, max_value=128, step=16)
    hp_layer_3 = hp.Int('layer_3', min_value=16, max_value=128, step=16)
    hp_layer_4 = hp.Int('layer_4', min_value=16, max_value=128, step=16)
    hp_optimiser = hp.Choice('optimiser', values=['adam', 'rmsprop'])
    hp_output = hp.Choice('output', values=['softmax', 'softplus'])
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.05)
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.05)
    hp_dropout_3 = hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.05)

    model.add(Dense(units=hp_layer_1,
                    kernel_initializer='lecun_uniform', input_shape=(27,), activation=hp_activation))
    model.add(Dense(units=hp_layer_2,
                     kernel_initializer='lecun_uniform', activation=hp_activation))
    model.add(Dropout(rate=hp_dropout_1))
    model.add(Dense(units=hp_layer_3,
                     kernel_initializer='lecun_uniform', activation=hp_activation))
    model.add(Dropout(hp_dropout_2))
    model.add(Dense(units=hp_layer_4,
                     kernel_initializer='lecun_uniform', activation=hp_activation))
    model.add(Dropout(hp_dropout_3))
    model.add(Dense(3, kernel_initializer='lecun_uniform', activation=hp_output))
    model.compile(loss="categorical_crossentropy", optimizer=hp_optimiser, metrics=[keras.metrics.AUC()])
    model.summary()
    return model
"""""
def prec_class1(y_true, y_pred):
    from sklearn.metrics import precision_recall_curve
    threshold = 0.76
    y_pred = np.squeeze(y_pred, axis=1)
    y_true = np.squeeze(y_true, axis=1)
    precision, recall, _ = precision_recall_curve(y_true[:, 1], y_pred[:, 1])
    for m in range(len(recall)):
        if recall[m] > threshold and recall[m] < threshold + 0.001:
            prec = precision[m]
    return threshold

class MyTuner(kt.Tuner):

    def run_trial(self, trial, y_true, y_pred):
        model = self.hypermodel.build(trial.hyperparameters)
        score = prec_class1(y_true, y_pred)
        self.oracle.update_trial(trial.trial_id, {'score': score})
        self.oracle.save_model(trial.trial_id, model)


#define tuner
tuner = kt.RandomSearch(model_builder,
                     objective=kt.Objective('auc', direction='max'),
                     max_trials=5,
                     directory='/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/dir',
                     project_name='PRISMA_Hyperparams'
                     )
"""""
callback = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    verbose=0,
    mode="auto",
)

#define tuner
tuner = kt.RandomSearch(model_builder,
                     objective=kt.Objective('auc', direction='max'),
                     max_trials=50,
                     directory='/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/dir',
                     project_name='PRISMA_Hyperparams'
                     )

# Load and format data
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")
x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))
Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)
x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2)

"""""
tuner.search(x_train, y_train,
             epochs=30,
             validation_data=(x_test, y_test),
             callbacks=[callback]
             )
tuner.results_summary()
"""""


# Return all trials from the oracle
tuner.reload()
trials = tuner.oracle.trials

# Print out the ID and the score of all trials
for trial_id, trial in trials.items():
    print(trial_id, trial.score)

# Return trials, append VH and WWW scores to list
# Note this is necessary because the 'best trials' will just have the best auc score
# --> They will not necessarily have the best S/B ratio
VH_over_bkg = []
WWW_over_bkg = []

# Can adjust number of trials churned through
best_trials = tuner.oracle.get_best_trials(num_trials=20)
print(best_trials)
for trial in best_trials:
    trial_id = trial.trial_id
    #trial.summary()
    model = tuner.load_model(trial)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    y_pred = np.concatenate((y_pred_train, y_pred_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    VH_over_bkg.append([trial_id, sigoverbkg('VH', y_pred, y, name_trial=trial_id)])
    WWW_over_bkg.append([trial_id, sigoverbkg('WWW', y_pred, y, name_trial=trial_id)])

# Sort lists so higher S/B are the first items
VH_over_bkg = sorted(VH_over_bkg, key=lambda x:x[1], reverse=True)
WWW_over_bkg = sorted(WWW_over_bkg, key=lambda x:x[1], reverse=True)

# Save the top 5 trials for VH and WWW
for j in range(0, 5):
    best_VH_id = VH_over_bkg[j][0]
    trial_VH = tuner.oracle.trials[best_VH_id]
    trial_VH.summary()
    best_WWW_id = WWW_over_bkg[j][0]
    trial_WWW = tuner.oracle.trials[best_WWW_id]
    trial_WWW.summary()

    model_VH = tuner.load_model(trial_VH)
    model_WWW = tuner.load_model(trial_WWW)

    filename_VH = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/VH_tuned_model_{j}.sav'
    filename_WWW = f'/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/WWW_tuned_model_{j}.sav'
    joblib.dump(model_VH, filename_VH)
    joblib.dump(model_WWW, filename_WWW)