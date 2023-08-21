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

N_lhe_sig = 1000000  # getNeventsLHE(lhe_signal)
N_lhe_bkg = 4000000  # getNeventsLHE(lhe_bkg)

N_root_sig = 511723  # getNeventsRoot(root_signal)
N_root_bgk = 2964674  # getNeventsRoot(root_bkg)


# calculate resulting xSection*efficiency
xSection_sig *= N_root_sig / N_lhe_sig
xSection_bkg *= N_root_bgk / N_lhe_bkg

# scale to given luminosity
lumi = 400  # inverse fb

n_sig = lumi * xSection_sig * 1000
n_bkg = lumi * xSection_bkg * 1000

weight_sig = n_sig / N_root_sig
weight_bkg = n_bkg / N_root_bgk

featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met', 'delR_l0l1', 'delR_l0l2',
                'delR_l1l2', 'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2', 'dPhi_MET_l0',
                'dPhi_MET_l1', 'dPhi_MET_l2', 'max_PT_jet', 'd0_l0', 'd0_l1', 'd0_l2',
                'z0sintheta_l0', 'z0sintheta_l1', 'z0sintheta_l2', 'n_btag', 'mT_l0l1',
                'mT_l0l2', 'mT_l1l2', 'sumPT', 'm_l0l1', 'm_l0l2', 'm_l1l2', 'm_lll', 'F_alpha']


plothisttriple(df_VH, df_WWW, df_bkg, ["PT_l0", "PT_l1", "PT_l2"], 400., 800., 1500., 10, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH, df_WWW, df_bkg, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 1500., 2000., 20, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH, df_WWW, df_bkg, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 2.e5, 6.e5, 8.e5, 16, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH, df_WWW, df_bkg, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4.1, 0.9, 1.3, 8, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH, df_WWW, df_bkg, ["d0_l0", "d0_l1", "d0_l2"], 1., float(np.max(df["d0_l1"])), float(np.max(df["d0_l2"])), 10, weight_sig, weight_bkg, lumi)
plothisttriple(df_VH, df_WWW, df_bkg, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., float(np.max(df["delR_l0l2"])), float(np.max(df["delR_l1l2"])), 30, weight_sig, weight_bkg, lumi)
plothist(df_VH, df_WWW, df_bkg, "m_lll", 0., 1500., 20, weight_sig, weight_bkg, lumi)
plothist(df_VH, df_WWW, df_bkg, "sumPT", 0., 1500., 100, weight_sig, weight_bkg, lumi)
plothist(df_VH, df_WWW, df_bkg, "n_btag", 0., 10., 9, weight_sig, weight_bkg, lumi)
plothist(df_VH, df_WWW, df_bkg, "max_PT_jet", 20., 200., 20, weight_sig, weight_bkg,lumi)
plothist(df_VH, df_WWW, df_bkg, "met", 0., 500., 100, weight_sig, weight_bkg, lumi)
plothist(df_VH, df_WWW, df_bkg, "F_alpha", 0., .0065, 20, weight_sig, weight_bkg, lumi)
