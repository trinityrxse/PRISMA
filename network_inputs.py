import uproot, pylhe, glob, os
import numpy as np
import numba, vector
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from analysis import *

lhe_signal = glob.glob(
    "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].lhe.gz")
root_signal = glob.glob(
    "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].root")

lhe_bkg = glob.glob("/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.lhe.gz")
root_bkg = glob.glob("/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.root")

df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed.csv")

df_VH = df[df['Type'] == 0]
df_WWW = df[df['Type'] == 1]
df_bkg = df[df['Type'] == 2]

xSection_sig = get_xSection(lhe_signal)
xSection_bkg = get_xSection(lhe_bkg)
print(xSection_sig, xSection_bkg)

N_lhe_sig = 1000000  # getNeventsLHE(lhe_signal)
N_lhe_bkg = 4000000  # getNeventsLHE(lhe_bkg)
print(N_lhe_sig, N_lhe_bkg)

N_root_sig = 511723  # getNeventsRoot(root_signal)
N_root_bgk = 2964674  # getNeventsRoot(root_bkg)
print(N_root_sig, N_root_bgk)

# calculate resulting xSection*efficiency
xSection_sig *= N_root_sig / N_lhe_sig
xSection_bkg *= N_root_bgk / N_lhe_bkg
print(xSection_sig, xSection_bkg)

# scale to given luminosity
lumi = 400  # inverse fb

n_sig = lumi * xSection_sig * 1000
n_bkg = lumi * xSection_bkg * 1000

weight_sig = n_sig / N_root_sig
weight_bkg = n_bkg / N_root_bgk

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
bins = np.linspace(0., 500., 100)
ax[0].hist([df_VH['met'], df_WWW['met'], df_bkg['met']], bins=bins,
           weights=[np.full(df_VH['met'].shape, weight_sig),
                    np.full(df_WWW['met'].shape, weight_sig),
                    np.full(df_bkg['met'].shape, weight_bkg)],
           histtype='bar', stacked=True, label=["VH", "WWW", "bkg"])
ax[0].set_yscale('log')
ax[0].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[0].transAxes)
ax[0].set_xlabel("MET [GeV]")
ax[0].set_ylabel("Entries")
ax[0].legend()

bins = np.linspace(0., 4., 80)
ax[1].hist([df_VH['delR_l0l1'], df_WWW['delR_l0l1'], df_bkg['delR_l0l1']], bins=bins,
           weights=[np.full(df_VH['delR_l0l1'].shape, weight_sig),
                    np.full(df_WWW['delR_l0l1'].shape, weight_sig),
                    np.full(df_bkg['delR_l0l1'].shape, weight_bkg)],
           histtype='step', stacked=False, density=True, label=["VH", "WWW", "bkg"])
ax[1].set_yscale('log')
ax[1].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[1].transAxes)
ax[1].set_xlabel("min($\\Delta R_{\\ell\\ell})$")
ax[1].set_ylabel("1/N dN/d min($\\Delta R_{\\ell\\ell})$")
ax[1].legend()

#plt.show()

featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met', 'delR_l0l1', 'delR_l0l2',
                'delR_l1l2', 'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2', 'dPhi_MET_l0',
                'dPhi_MET_l1', 'dPhi_MET_l2', 'max_PT_jet', 'd0_l0', 'd0_l1', 'd0_l2',
                'z0sintheta_l0', 'z0sintheta_l1', 'z0sintheta_l2', 'n_btag', 'mT_l0l1',
                'mT_l0l2', 'mT_l1l2', 'sumPT', 'm_l0l1', 'm_l0l2', 'm_l1l2', 'm_lll']

def plothist(feature, binstart, binend, numberbins):
    featurename = str(feature)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(binstart, binend, numberbins)
    ax.hist([df_VH[feature], df_WWW[feature], df_bkg[feature]], bins=bins,
            weights=[np.full(df_VH[feature].shape, weight_sig),
                     np.full(df_WWW[feature].shape, weight_sig),
                     np.full(df_bkg[feature].shape, weight_bkg)],
            histtype='bar', stacked=True, label=["VH", "WWW", "bkg"])
    ax.set_yscale('log')
    ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
    ax.set_xlabel(featurename)
    ax.set_ylabel("Entries")
    ax.legend()
    plt.show()

#plothist("m_l0l2", 0, 1500, 20)
#plothist("m_l0l1", 0, 300, 20)
#plothist("m_l1l2", 0, 2000, 20)
#plothist("m_lll", 0, 1500, 20)
#plothist("sumPT", 0, 1500, 100)
#plothist("mT_l0l2", 0, 6e5, 20)
#plothist("mT_l0l1", 0, 2e5, 20)
#plothist("mT_l1l2", 0, 8e5, 20)
#plothist("n_btag", 0, 10, 9)
#plothist("z0sintheta_l2", 0, 1.3, 10)
#plothist("z0sintheta_l1", 0, 0.9, 10)
#plothist("z0sintheta_l0", 0, 4.1, 10)
#plothist("d0_l0", 0., 1., 10)
#plothist("d0_l1", 0, float(np.max(df["d0_l1"])), 10)
#plothist("d0_l2", 0, float(np.max(df["d0_l2"])), 10)
#plothist("max_PT_jet", 20, 200, 20)
#plothist("delR_l0l1", 1, 4, 20)
#plothist("delR_l0l2", 1, float(np.max(df["delR_l0l2"])), 20)
#plothist("delR_l1l2", 1, float(np.max(df["delR_l1l2"])), 20)
#plothist("met", 0., 500., 100)
#plothist("PT_l0", 0, 400, 10)
#plothist("PT_l1", 0, 800, 10)
#plothist("PT_l2", 0, 1500, 10)

def plothisttriple(featurelist, binstart0, binend0, binstart1, binend1, binstart2, binend2, numberbins):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    bins0 = np.linspace(binstart0, binend0, numberbins)
    featurename0 = str(featurelist[0])
    ax[0].hist([df_VH[featurelist[0]], df_WWW[featurelist[0]], df_bkg[featurelist[0]]], bins=bins0,
            weights=[np.full(df_VH[featurelist[0]].shape, weight_sig),
                     np.full(df_WWW[featurelist[0]].shape, weight_sig),
                     np.full(df_bkg[featurelist[0]].shape, weight_bkg)],
            histtype='bar', stacked=True, label=["VH", "WWW", "bkg"])
    ax[0].set_yscale('log')
    ax[0].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[0].transAxes)
    ax[0].set_xlabel(featurename0)
    ax[0].set_ylabel("Entries")
    ax[0].legend()

    bins1 = np.linspace(binstart1, binend1, numberbins)
    featurename1 = str(featurelist[1])
    ax[1].hist([df_VH[featurelist[1]], df_WWW[featurelist[1]], df_bkg[featurelist[1]]], bins=bins1,
               weights=[np.full(df_VH[featurelist[1]].shape, weight_sig),
                        np.full(df_WWW[featurelist[1]].shape, weight_sig),
                        np.full(df_bkg[featurelist[1]].shape, weight_bkg)],
               histtype='bar', stacked=True, label=["VH", "WWW", "bkg"])
    ax[1].set_yscale('log')
    ax[1].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[1].transAxes)
    ax[1].set_xlabel(featurename1)
    ax[1].set_ylabel("Entries")
    ax[1].legend()

    bins2 = np.linspace(binstart2, binend2, numberbins)
    featurename2 = str(featurelist[2])
    ax[2].hist([df_VH[featurelist[2]], df_WWW[featurelist[2]], df_bkg[featurelist[2]]], bins=bins0,
               weights=[np.full(df_VH[featurelist[2]].shape, weight_sig),
                        np.full(df_WWW[featurelist[2]].shape, weight_sig),
                        np.full(df_bkg[featurelist[2]].shape, weight_bkg)],
               histtype='bar', stacked=True, label=["VH", "WWW", "bkg"])
    ax[2].set_yscale('log')
    ax[2].text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax[2].transAxes)
    ax[2].set_xlabel(featurename2)
    ax[2].set_ylabel("Entries")
    ax[2].legend()

    plt.show()

plothisttriple(["PT_l0", "PT_l1", "PT_l2"], 0, 2000, 0, 2000, 0, 2000, 10)