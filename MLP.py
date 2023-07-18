import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
# Train-Test
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# Scaling data
from sklearn.preprocessing import StandardScaler
# Classification Report
from sklearn.metrics import classification_report
from keras.utils import to_categorical

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
class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
print(class_weight)
class_weight = {0: 5.59882475,
                1: 4.76806281,
                2: 0.3828979}
callback = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    verbose=0,
    mode="auto",
)

model = Sequential()
model.add(Dense(34, input_shape = (29,), activation = "swish"))
model.add(Dense(80, activation = "swish"))
model.add(Dropout(0.37))
model.add(Dense(32, activation = "swish"))
model.add(Dropout(0.3))
model.add(Dense(16, activation = "swish"))
model.add(Dropout(0.26))
model.add(Dense(3, activation = "softmax"))
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics = ["accuracy"])
model.summary()

model.fit(x_train, y_train, verbose=1, epochs=2, class_weight = class_weight, callbacks=[callback])

pred_train = model.predict(x_train)

scores = model.evaluate(x_train, y_train, verbose=0)

print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model.predict(x_test)

scores2 = model.evaluate(x_test, y_test, verbose=0)

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
print(y_pred)
y_test_class = np.argmax(y_test, axis=1)
#confusion_matrix(y_test_class, y_pred_class)

print(classification_report(y_test_class, y_pred_class))


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = ['VH', 'WWW', 'bkg']
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def main():
    sampleClassificationReport =     """"" precision    recall  f1-score   support

           0       0.19      0.87      0.31     20020
           1       0.28      0.76      0.41     23172
           2       0.98      0.61      0.75    292382 """""

    plot_classification_report(sampleClassificationReport)
    plt.close()

if __name__ == "__main__":
    main()
    plt.savefig('/localscratch/classif_report.png', dpi=200, format='png', bbox_inches='tight')
    #cProfile.run('main()') # if you want to do some profiling
