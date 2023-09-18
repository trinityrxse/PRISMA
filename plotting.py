
'''

Plotting the histograms of input variables and how the 
        variables change after being passed through networks
        or how they differ for each EFT

'''
import numpy as np
import matplotlib.pyplot as plt
from selection_criteria import *



def plothist(filepath, df, featurelist, binend, numberbins, df2=None, weight2=None, name2=None):
    #plots distributions for input data
    #if df2 is not None, can be used to plot EFT inputs, or output distributions post ML
    weight_sig, weight_bkg, lumi = get_weights(filepath)

    classes = len(featurelist)

    df_VH = df[df['Type'] == 0]
    df_WWW = df[df['Type'] == 1]
    df_bkg = df[df['Type'] == 2]

    if classes == 1:
        featurename = str(featurelist[0])
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bins = np.linspace(0., binend, numberbins)

        if df2 is not None:
            # df for the 3 categories
            df_VH2 = df2[df2['Type'] == 0]
            df_WWW2 = df2[df2['Type'] == 1]
            df_bkg2 = df2[df2['Type'] == 2]
            ax.hist([df_VH2[featurelist[0]], df_WWW2[featurelist[0]], df_bkg2[featurelist[0]]], color=['red', 'yellow', 'blue'],
                    bins=bins,
                    weights=[np.full(df_VH2[featurelist[0]].shape, weight2),
                             np.full(df_WWW2[featurelist[0]].shape, weight2),
                             np.full(df_bkg2[featurelist[0]].shape, weight_bkg)],
                    histtype='step',
                    label=[f"VH {name2}", f"WWW {name2}", f"bkg {name2}"])

        ax.hist([df_VH[featurelist[0]], df_WWW[featurelist[0]], df_bkg[featurelist[0]]], color=['purple', 'lime', 'cyan'],
                bins=bins,
                weights=[np.full(df_VH[featurelist[0]].shape, weight_sig),
                         np.full(df_WWW[featurelist[0]].shape, weight_sig),
                         np.full(df_bkg[featurelist[0]].shape, weight_bkg)],
                histtype='step', label=["VH", "WWW", "bkg"])
        ax.set_yscale('log')
        ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
        ax.set_xlabel(featurename)
        ax.set_ylabel("Entries")
        ax.legend()
        plt.tight_layout()
        plt.show()

    elif classes == 2:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        bins = np.linspace(0, binend, numberbins)
        featurename0 = str(featurelist[0])

        if df2 is not None:
            # df for the 3 categories
            df_VH2 = df2[df2['Type'] == 0]
            df_WWW2 = df2[df2['Type'] == 1]
            df_bkg2 = df2[df2['Type'] == 2]

            ax[0].hist([df_VH2[featurelist[0]], df_WWW2[featurelist[0]], df_bkg2[featurelist[0]]], color=['red', 'yellow', 'blue'],
                       bins=bins,
                       weights=[np.full(df_VH2[featurelist[0]].shape, weight2),
                                np.full(df_WWW2[featurelist[0]].shape, weight2),
                                np.full(df_bkg2[featurelist[0]].shape, weight2)],
                       histtype='step', label=[f"VH {name2}", f"WWW {name2}", f"bkg {name2}"])
            ax[1].hist([df_VH2[featurelist[1]], df_WWW2[featurelist[1]], df_bkg2[featurelist[1]]],
                       color=['red', 'yellow', 'blue'],
                       bins=bins,
                       weights=[np.full(df_VH2[featurelist[0]].shape, weight2),
                                np.full(df_WWW2[featurelist[0]].shape, weight2),
                                np.full(df_bkg2[featurelist[0]].shape, weight_bkg)],
                       histtype='step', label=[f"VH {name2}", f"WWW {name2}", f"bkg {name2}"])

        ax[0].hist([df_VH[featurelist[0]], df_WWW[featurelist[0]], df_bkg[featurelist[0]]],
                   color=['purple', 'lime', 'cyan'],
                   bins=bins,
                   weights=[np.full(df_VH[featurelist[0]].shape, weight_sig),
                            np.full(df_WWW[featurelist[0]].shape, weight_sig),
                            np.full(df_bkg[featurelist[0]].shape, weight_bkg)],
                   histtype='step', label=["VH", "WWW", "bkg"])
        ax[0].set_yscale('log')
        ax[0].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[0].transAxes)
        ax[0].set_xlabel(featurename0)
        ax[0].set_ylabel("Entries")
        ax[0].legend()

        featurename1 = str(featurelist[1])
        ax[1].hist([df_VH[featurelist[1]], df_WWW[featurelist[1]], df_bkg[featurelist[1]]],
                   color=['purple', 'lime', 'cyan'],
                   bins=bins,
                   weights=[np.full(df_VH[featurelist[1]].shape, weight_sig),
                            np.full(df_WWW[featurelist[1]].shape, weight_sig),
                            np.full(df_bkg[featurelist[1]].shape, weight_bkg)],
                   histtype='step', label=["VH", "WWW", "bkg"])
        ax[1].set_yscale('log')
        ax[1].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[1].transAxes)
        ax[1].set_xlabel(featurename1)
        ax[1].set_ylabel("Entries")
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    elif classes == 3:
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        bins = np.linspace(0, binend, numberbins)
        featurename0 = str(featurelist[0])

        if df2 is not None:
            # df for the 3 categories
            df_VH2 = df2[df2['Type'] == 0]
            df_WWW2 = df2[df2['Type'] == 1]
            df_bkg2 = df2[df2['Type'] == 2]

            ax[0].hist([df_VH2[featurelist[0]], df_WWW2[featurelist[0]], df_bkg2[featurelist[0]]],
                       color=['red', 'yellow', 'blue'],
                       bins=bins,
                       weights=[np.full(df_VH2[featurelist[0]].shape, weight2),
                                np.full(df_WWW2[featurelist[0]].shape, weight2),
                                np.full(df_bkg2[featurelist[0]].shape, weight_bkg)],
                       histtype='step', label=[f"VH {name2}", f"WWW {name2}", f"bkg {name2}"])
            ax[1].hist([df_VH2[featurelist[1]], df_WWW2[featurelist[1]], df_bkg2[featurelist[1]]],
                       color=['red', 'yellow', 'blue'],
                       bins=bins,
                       weights=[np.full(df_VH2[featurelist[0]].shape, weight2),
                                np.full(df_WWW2[featurelist[0]].shape, weight2),
                                np.full(df_bkg2[featurelist[0]].shape, weight_bkg)],
                       histtype='step', label=[f"VH {name2}", f"WWW {name2}", f"bkg {name2}"])
            ax[2].hist([df_VH2[featurelist[2]], df_WWW2[featurelist[2]], df_bkg2[featurelist[2]]],
                       color=['red', 'yellow', 'blue'],
                       bins=bins,
                       weights=[np.full(df_VH2[featurelist[0]].shape, weight2),
                                np.full(df_WWW2[featurelist[0]].shape, weight2),
                                np.full(df_bkg2[featurelist[0]].shape, weight_bkg)],
                       histtype='step', label=[f"VH {name2}", f"WWW {name2}", f"bkg {name2}"])

        ax[0].hist([df_VH[featurelist[0]], df_WWW[featurelist[0]], df_bkg[featurelist[0]]],
                   color=['purple', 'lime', 'cyan'],
                   bins=bins,
                   weights=[np.full(df_VH[featurelist[0]].shape, weight_sig),
                            np.full(df_WWW[featurelist[0]].shape, weight_sig),
                            np.full(df_bkg[featurelist[0]].shape, weight_bkg)],
                   histtype='step', label=["VH", "WWW", "bkg"])
        ax[0].set_yscale('log')
        ax[0].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[0].transAxes)
        ax[0].set_xlabel(featurename0)
        ax[0].set_ylabel("Entries")
        ax[0].legend()

        featurename1 = str(featurelist[1])
        ax[1].hist([df_VH[featurelist[1]], df_WWW[featurelist[1]], df_bkg[featurelist[1]]],
                   color=['purple', 'lime', 'cyan'],
                   bins=bins,
                   weights=[np.full(df_VH[featurelist[1]].shape, weight_sig),
                            np.full(df_WWW[featurelist[1]].shape, weight_sig),
                            np.full(df_bkg[featurelist[1]].shape, weight_bkg)],
                   histtype='step', label=["VH", "WWW", "bkg"])
        ax[1].set_yscale('log')
        ax[1].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[1].transAxes)
        ax[1].set_xlabel(featurename1)
        ax[1].set_ylabel("Entries")
        ax[1].legend()

        featurename2 = str(featurelist[2])
        ax[2].hist([df_VH[featurelist[2]], df_WWW[featurelist[2]], df_bkg[featurelist[2]]], color=['purple', 'lime', 'cyan'],
                   bins=bins,
                   weights=[np.full(df_VH[featurelist[2]].shape, weight_sig),
                            np.full(df_WWW[featurelist[2]].shape, weight_sig),
                            np.full(df_bkg[featurelist[2]].shape, weight_bkg)],
                   histtype='step', label=["VH", "WWW", "bkg"])
        ax[2].set_yscale('log')
        ax[2].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[2].transAxes)
        ax[2].set_xlabel(featurename2)
        ax[2].set_ylabel("Entries")
        ax[2].legend()
        plt.tight_layout()
        plt.show()