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

N_lhe_sig = 1000000  # getNeventsLHE(lhe_signal)
N_lhe_bkg = 4000000  # getNeventsLHE(lhe_bkg)
print(N_lhe_sig, N_lhe_bkg)

N_root_sig = 511723  # getNeventsRoot(root_signal)
N_root_bgk = 2964674  # getNeventsRoot(root_bkg)
print(N_root_sig, N_root_bgk)

# show available branches
tree = uproot.open(root_signal[0] + ':Delphes')
tree.keys()
tree['Electron.ErrorD0'].arrays()

ntotal_VH = 0
nselected_VH = 0
df_VH = None

myfilter = "/(MissingET.(MET|Phi)|Particle.(PID|Status)|Electron.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Muon.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Jet.(PT|Eta|Phi|BTag))/"
batch_size = 5000
iterator = uproot.iterate([f + ":Delphes" for f in root_signal], step_size=batch_size, filter_name=myfilter)

for batch in tqdm(iterator, total=N_root_sig // batch_size):
    higgs = checkHiggs(batch)
    presel = preselection(batch)
    ntotal_VH += higgs.sum()
    nselected_VH += np.logical_and(higgs, presel).sum()

    batch = batch_selector(batch, 0)  # VH data

    PT_l0, PT_l1, PT_l2, met, \
    dR_l0l1, dR_l0l2, dR_l1l2, \
    deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
    transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
    invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
    dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
    d0_l0, d0_l1, d0_l2, n_jet, \
    n_btag, zsin_l0, zsin_l1, zsin_l2 = get_leptons(batch)
    # remaining is F_alpha and cannot compute max(1 - E/p)

    dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
               'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
               'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
               'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2,
               'max_PT_jet': max_PT_jet,
               'd0_l0': d0_l0, 'd0_l1': d0_l1, 'd0_l2': d0_l2,
               'z0sintheta_l0': zsin_l0, 'z0sintheta_l1': zsin_l1, 'z0sintheta_l2': zsin_l2, 'n_btag': n_btag,
               'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
               'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2': invarM_l1l2, 'm_lll': invarM}
    # make into dict

    if df_VH is not None:
        newdata_VH = pd.DataFrame(dataset)
        df_VH = pd.concat([df_VH, newdata_VH], axis=0, ignore_index=True)

    else:
        df_VH = pd.DataFrame(dataset)

print("Preselection efficiency for VH events: %2.2f %%" % (100. * nselected_VH / ntotal_VH))
df_VH.insert(loc=0, column='Type', value=0)

ntotal_WWW = 0
nselected_WWW = 0
df_WWW = None

myfilter = "/(MissingET.(MET|Phi)|Particle.(PID|Status)|Electron.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Muon.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Jet.(PT|Eta|Phi|BTag))/"
batch_size = 5000
iterator = uproot.iterate([f + ":Delphes" for f in root_signal], step_size=batch_size, filter_name=myfilter)

for batch in tqdm(iterator, total=N_root_sig // batch_size):
    higgs = checkHiggs(batch)
    presel = preselection(batch)
    ntotal_WWW += higgs.sum()
    nselected_WWW += np.logical_and(higgs, presel).sum()

    batch = batch_selector(batch, 1)  # WWW data

    PT_l0, PT_l1, PT_l2, met, \
    dR_l0l1, dR_l0l2, dR_l1l2, \
    deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
    transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
    invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
    dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
    d0_l0, d0_l1, d0_l2, n_jet, \
    n_btag, zsin_l0, zsin_l1, zsin_l2 = get_leptons(batch)
    # remaining is F_alpha and cannot compute max(1 - E/p)

    dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
               'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
               'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
               'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2,
               'max_PT_jet': max_PT_jet,
               'd0_l0': d0_l0, 'd0_l1': d0_l1, 'd0_l2': d0_l2,
               'z0sintheta_l0': zsin_l0, 'z0sintheta_l1': zsin_l1, 'z0sintheta_l2': zsin_l2, 'n_btag': n_btag,
               'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
               'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2': invarM_l1l2, 'm_lll': invarM}
    # make into dict

    if df_WWW is not None:
        newdata_WWW = pd.DataFrame(dataset)
        df_WWW = pd.concat([df_WWW, newdata_WWW], axis=0, ignore_index=True)

    else:
        df_WWW = pd.DataFrame(dataset)
print("Preselection efficiency for WWW events: %2.2f %%" % (100. * nselected_WWW / ntotal_WWW))

df_WWW.insert(loc=0, column='Type', value=1)

# loop over background
ntotal = 0
nselected = 0
df_bkg = None
batch_size = 5000
iterator = uproot.iterate([f + ":Delphes" for f in root_bkg], step_size=batch_size, filter_name=myfilter)
for batch in tqdm(iterator, total=N_root_bgk / batch_size):
    higgs = checkHiggs(batch)
    presel = preselection(batch)
    batch = batch_selector(batch, None)
    PT_l0, PT_l1, PT_l2, met, \
    dR_l0l1, dR_l0l2, dR_l1l2, \
    deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
    transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
    invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
    dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
    d0_l0, d0_l1, d0_l2, n_jet, \
    n_btag, zsin_l0, zsin_l1, zsin_l2 = get_leptons(batch)
    # remaining is F_alpha and cannot compute max(1 - E/p)

    dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
               'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
               'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
               'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2,
               'max_PT_jet': max_PT_jet,
               'd0_l0': d0_l0, 'd0_l1': d0_l1, 'd0_l2': d0_l2,
               'z0sintheta_l0': zsin_l0, 'z0sintheta_l1': zsin_l1, 'z0sintheta_l2': zsin_l2, 'n_btag': n_btag,
               'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
               'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2': invarM_l1l2, 'm_lll': invarM}
    # make into dict
    # any round but first round
    if df_bkg is not None:
        newdata_bkg = pd.DataFrame(dataset)
        df_bkg = pd.concat([df_bkg, newdata_bkg], axis=0, ignore_index=True)

    # first round
    else:
        df_bkg = pd.DataFrame(dataset)

        ntotal += len(batch)
        nselected += presel.sum()
print("Preselection efficiency for background events: %2.2f %%" % (100. * nselected / ntotal))
df_bkg.insert(loc=0, column='Type', value=2)
df_bkg.head()

df_p = pd.concat([df_WWW, df_VH, df_bkg], axis=0, ignore_index=True)
df_p.shape
df_p.head()
df_p.to_csv(r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed.csv", index=False)

corr_df = df_p.corr()

corr_df.dropna()

sns.heatmap(corr_df,
            xticklabels=corr_df.columns,
            yticklabels=corr_df.columns,
            )
plt.savefig("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/corro_plot.png")
plt.show()
