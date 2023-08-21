import keras
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

def KFold_CV(df, model):
    # Load dataset - can change to EFT samples
    x = df.drop('Type', axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))

    Y = df['Type'].values
    y_cat = to_categorical(Y, num_classes=3)

    # Load trained ML model
    model.summary()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
    class_weight = {0: class_weight[0],
                    1: class_weight[1],
                    2: class_weight[2]}
    callback = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=3,
        verbose=0,
        mode="auto"
    )

    weights_init = model.get_weights()
    validation_scores = []
    featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met', 'delR_l0l1', 'delR_l0l2',
                    'delR_l1l2', 'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2', 'dPhi_MET_l0',
                    'dPhi_MET_l1', 'dPhi_MET_l2',
                    'z0sintheta_l0', 'n_btag', 'mT_l0l1',
                    'mT_l0l2', 'mT_l1l2', 'sumPT', 'm_l0l1', 'm_l0l2', 'm_lll', 'F_alpha']

    df_ML = None
    for i, (train_index, test_index) in enumerate(skf.split(x, Y)):
        print(f"Fold {i}:")
        print(f" Train:index={train_index}")
        print(f" Test:index={test_index}")
        x = np.array(x)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_cat[train_index], y_cat[test_index]

        model.set_weights(weights_init)
        model.fit(x_train, y_train, verbose=1, epochs=100, class_weight=class_weight, callbacks=[callback])
        validation_score = model.evaluate(x_test, y_test)
        print(validation_score)
        validation_scores.append(validation_score)

        y_test_class = np.argmax(y_test, axis=1)  # classes predicted for test data using model
        x_test = np.array(sc.inverse_transform(x_test))

        if df_ML is not None:
            newdata = pd.DataFrame(x_test, columns=featurenames)
            newdata.insert(loc=0, column='Type', value=y_test_class)
            df_ML = pd.concat([df_ML, newdata], axis=0, ignore_index=True)
        else:
            df_ML = pd.DataFrame(x_test, columns=featurenames)
            df_ML.insert(loc=0, column='Type', value=y_test_class)


    avg = np.mean(np.array(validation_scores)[:, 1])
    print("Average validation accuracy ", avg)

    return df_ML