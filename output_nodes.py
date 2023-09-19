
'''

Acquiring output nodes 
Plotting output nodes
Acquiring delta nodes

'''

import numba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from selection_criteria import *


def get_multiclassifier_nodes(filepath, y_array, y_pred):
    y_pred_VH = [i[0] for i in y_pred]
    y_array_VH = [i[0] for i in y_array]
    y_pred_WWW = [i[1] for i in y_pred]
    y_array_WWW = [i[1] for i in y_array]
    y_pred_bkg = [i[2] for i in y_pred]
    y_array_bkg = [i[2] for i in y_array]
    results = {"truth 0": y_pred_VH,
               "prediction 0": y_array_VH,
               "truth 1": y_pred_WWW,
               "prediction 1": y_array_WWW,
               "truth 2": y_pred_bkg,
               "prediction 2": y_array_bkg,
               }
    df_test = pd.DataFrame(results)

    weight_sig, weight_bkg, lumi = get_weights(filepath)
    conditions = [
        (df_test["truth 0"] == 1.0),
        (df_test["truth 1"] == 1.0),
        (df_test["truth 2"] == 1.0)]
    choices = [weight_sig, weight_sig, weight_bkg]
    df_test['weights'] = np.select(conditions, choices, default=0)

    return df_test

@numba.jit(nopython=True)
def get_EFT_nodes(y_test, y_pred):

    SM_dist = []
    EFT_dist = []
    for i in range(0, len(y_pred)):
        if y_test[i] == 1:
            SM_dist.append(y_pred[i])
        elif y_test[i] == 0:
            EFT_dist.append(y_pred[i])

    return SM_dist, EFT_dist

def plot_nodes_EFT(filepath, y_test, y_pred, weightEFT, name, ret=False):
    SM_dist, EFT_dist = get_EFT_nodes(y_test, y_pred)

    weight_sig, weight_bkg, lumi = get_weights(filepath)  # weights & luminosity

    names = ["SM", "EFT"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(0, 1, 40)
    ax.hist([SM_dist, EFT_dist], color=['purple', 'cyan'], bins=bins,
            weights=[np.full(len(SM_dist), weight_sig),
                     np.full(len(EFT_dist), weightEFT)],
            histtype='step', linewidth=5, label=["truth SM", "truth EFT"])
    ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
    ax.set_xlabel(f"Network Output Distribution for Node")
    #ax.set_yscale("log")
    ax.set_ylabel("Entries")
    ax.legend()
    plt.savefig(
        f"{filepath}/SM_node_{name}.png")
    plt.show()

    if ret == True:
        return SM_dist, EFT_dist

def plot_nodes_multiclassifier(filepath, y_array, y_pred, ret=False):
    df_test = get_multiclassifier_nodes(filepath, y_array, y_pred)

    cutVH = df_test[df_test["truth 0"] == 1]
    cutWWW = df_test[df_test["truth 1"] == 1]
    cutbkg = df_test[df_test["truth 2"] == 1]

    array_list = []

    for i in range(0, 3):
        if i == 0:
            var = 'VH'
        elif i == 1:
            var = 'WWW'
        else:
            var = 'bkg'
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bins = np.linspace(0., 1, 20)

        weight_sig, weight_bkg, lumi = get_weights(filepath)
        array_list_X, bins_X, patches = ax.hist([cutVH[f"prediction {i}"],
                 cutWWW[f"prediction {i}"],
                 cutbkg[f"prediction {i}"]
                 ],
            weights = [cutVH[f"weights"],
                    cutWWW[f"weights"],
                    cutbkg[f"weights"]],
            color=['purple', 'lime', 'cyan'],
            bins=bins,
            histtype='step',
            label=["truth VH", "truth WWW", "truth bkg"])
        #ax.set_yscale('log')
        ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
        ax.set_xlabel(f"{var} Output Node")
        ax.set_ylabel("Entries")
        ax.legend(loc='upper left')
        plt.savefig(f'{filepath}/output_node_{var}')
        plt.show()

        array_list.append(array_list_X)

    if ret == True:
        return array_list

def s_over_b(filepath, y_array, y_pred, type, graph=False):
    # gets signal over background ratio
    # plots events in final bin ie correctly predicted signal over events pred
    #       as signal but no actually signal

    df_test = get_multiclassifier_nodes(filepath, y_pred, y_array)

    # whole df, sorted descending order of predicted type
    df1 = df_test.sort_values(f'prediction {type}', ascending=False, ignore_index=True)

    cut = (df1[f"truth {type}"] == 1.0)
    df2 = df1[cut] # df of all truth events of a type ie all pred VH that ARE VH

    #get number of events detected to 4
    sum_sig = 0
    index = 0
    for i in range(0, len(df2)):
        if sum_sig < 4:
            sum_sig += (df2.iloc[i][f"prediction {type}"]) * (df2.iloc[i]["weights"])
            index += 1

        else:
            continue

    if index >= 1:
        value = df2.iloc[index][f"prediction {type}"]

        sum_bkg = 0
        cut = (df1[f"truth {type}"] != 1.0) & (df1[f"prediction {type}"] > value)
        df3 = df1[cut] #df of all events identified as type but actually aren't


        bkg = (df3[f"prediction {type}"]) * (df3["weights"])
        sum_bkg = bkg.sum(axis='index')

        s_over_b = sum_sig / sum_bkg

        if graph is not False:
            if type == '0':
                label ='VH'
            elif type == '1':
                label = 'WWW'

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            bins = np.linspace(0.9, 1, 40)
            ax.hist([df2.iloc[0:index][f"prediction {type}"], df3.iloc[0:index][f"prediction {type}"]],
                    color=['red', 'blue'],
                    bins=bins,
                    histtype='step',
                    weights = [df2.iloc[0:index]["weights"], df3.iloc[0:index]["weights"]],
                    label=[f"Predicted {label}", f"Predicted not {label}"])
            ax.set_yscale('log')
            ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % 400, transform=ax.transAxes)
            ax.set_xlabel(f"{label} node")
            ax.set_ylabel("Entries")
            ax.legend()
            plt.show()

        return s_over_b
    
    else:

        return 'NaN'


def get_delta_node(filepath, y_array, y_pred):
    y_pred_VH = [i[0] for i in y_pred]
    y_array_VH = [i[0] for i in y_array]
    y_pred_WWW = [i[1] for i in y_pred]
    y_array_WWW = [i[1] for i in y_array]
    y_array_bkg = [i[2] for i in y_array]

    delta_node_VH = []
    delta_node_WWW = []
    delta_node_bkg = []

    for i in range(0, len(y_array)):
        if y_array_VH[i] == 1:
            delta_node_VH.append(y_pred_WWW[i] - y_pred_VH[i])
        if y_array_WWW[i] == 1:
            delta_node_WWW.append(y_pred_WWW[i] - y_pred_VH[i])
        if y_array_bkg[i] == 1: 
            delta_node_bkg.append(y_pred_WWW[i] - y_pred_VH[i])

    weight_sig, weight_bkg, lumi = get_weights(filepath)
    bins = np.linspace(-1, 1, 10)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist([delta_node_VH, delta_node_WWW, delta_node_bkg], color=['purple', 'lime', 'cyan'],
    bins=bins,
    weights=[np.full(len(delta_node_VH), weight_sig), 
            np.full(len(delta_node_WWW), weight_sig),
            np.full(len(delta_node_bkg), weight_bkg)],
    histtype='step', label=["Delta Node VH", "Delta Node WWW", "Delta Node Bkg"])
    #ax.set_yscale('log')
    ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
    ax.set_xlabel(f'Delta Node')
    ax.set_ylabel("Entries")
    ax.legend()
    plt.savefig(
        f"{filepath}/Delta_node.png")
    plt.tight_layout()
    plt.show()

    return delta_node_VH, delta_node_WWW, delta_node_bkg

def binned_delta_node(delta_node, weight):   
        data = delta_node
        bins = np.linspace(-1, 1, 10)
        digitized = np.digitize(data, bins)
        histogram_values = plt.hist(data, bins, weights=np.full(len(data), weight))[0]
        print(histogram_values)

        return histogram_values

def binned_EFT_node(EFT_node, weight):   
        data = EFT_node
        bins = np.linspace(0, 1, 10)
        digitized = np.digitize(data, bins)
        histogram_values = plt.hist(data, bins, weights=np.full(len(data), weight))[0]

        return histogram_values  