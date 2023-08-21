import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
# Make 3 classes
from keras.utils import to_categorical
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from analysis import *

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
    monitor="loss",
    patience=3,
    verbose=0,
    mode="auto",
)

def get_model(init_mode='lecun_uniform', activation='swish', output_act='softmax', optimizer='adam'):
    model = Sequential()
    model.add(Dense(27, kernel_initializer=init_mode, input_shape=(27,), activation=activation))
    model.add(Dense(34, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(0.37))
    model.add(Dense(34, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(34, kernel_initializer=init_mode, activation=activation))
    model.add(Dropout(0.26))
    model.add(Dense(3, kernel_initializer=init_mode, activation=output_act))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['AUC'])
    model.summary()
    return model




model = get_model()
model_history = model.fit(x_train, y_train, verbose=1, epochs=100, \
                          class_weight=class_weight, callbacks=[callback], validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
sigoverbkg_VH(y_pred, y_test)
sigoverbkg_WWW(y_pred, y_test)

# save the model to disk
filename = '/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/initial_model.sav'
joblib.dump(model, filename)

# Plot the loss function
plot_modelhistory(model_history, 'loss')
plot_modelhistory(model_history, 'auc')


