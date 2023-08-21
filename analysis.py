import uproot, pylhe, glob, os
import numpy as np
import numba,vector
import matplotlib.pyplot as plt
from keras import backend as K
from typing import List
import uproot, pylhe, glob, os
import numpy as np
import numba, vector
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import wandb
import joblib
# Classification Report
from sklearn.metrics import classification_report

def get_weights():
    lhe_signal = glob.glob(
        "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].lhe.gz")
    root_signal = glob.glob(
        "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].root")

    lhe_bkg = glob.glob("/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.lhe.gz")
    root_bkg = glob.glob("/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.root")

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

    return weight_sig, weight_bkg, lumi

def get_weights_EFT(EFTname, number_events):
    lhe_signal = glob.glob(
        f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_{EFTname}/WWW01j_00{number_events}.lhe.gz")
    root_signal = glob.glob(
        f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_{EFTname}/WWW01j_00{number_events}.root")

    xSection_sig = get_xSection(lhe_signal)

    N_lhe_sig = getNeventsLHE(lhe_signal)

    N_root_sig = getNeventsRoot(root_signal)

    # calculate resulting xSection*efficiency
    xSection_sig *= N_root_sig / N_lhe_sig

    # scale to given luminosity
    lumi = 400  # inverse fb

    n_sig = lumi * xSection_sig * 1000

    weight_sig = n_sig / N_root_sig

    return weight_sig
def get_xSection(lhefiles):
    init=pylhe.read_lhe_init(lhefiles[0])

    xSection=0.
    for process in init['procInfo']:
        xSection+=process['xSection']
    return xSection # in pb

def getNeventsLHE(lhefiles):
    N=0
    for f in lhefiles:
        lines=os.popen('zgrep "</event>" '+f+"|wc -l").readlines()
        N+=int(lines[0])
    return N

def getNeventsRoot(rootfiles):
    N=0
    for f in rootfiles:
        with uproot.open(f+':Delphes') as tree:
            N+=tree.num_entries
    return N

@numba.jit(nopython=True)
def checkHiggs(batch):
    higgs=np.full((len(batch),),False)
    for i in range(len(batch)):
        for p in range(len(batch[i]['Particle.PID'])):
            if abs(batch[i]['Particle.PID'][p])==25:
                higgs[i]=True
                break
    return higgs


@numba.jit(nopython=True)
def preselection(batch):
    pass_selection = np.full((len(batch),), False)
    # TODO make it need a unique charge
    for i in range(len(batch)):
        nElectrons = 0
        nMuons = 0
        nJets = 0
        n_sameQ = 0
        charges_e =[]
        charges_m = []

        for e in range(len(batch[i]['Electron.PT'])):
            if batch[i]['Electron.PT'][e] >= 10.0 and abs(batch[i]['Electron.Eta'][e]) < 2.5:
                nElectrons += 1

        for m in range(len(batch[i]['Muon.PT'])):
            if batch[i]['Muon.PT'][m] >= 10.0 and abs(batch[i]['Muon.Eta'][m]) < 2.5:
                nMuons += 1

        for j in range(len(batch[i]['Jet.PT'])):
            if batch[i]['Jet.PT'][j] >= 25.0 and abs(batch[i]['Jet.Eta'][j]) < 2.5:
                nJets += 1

        for e in range(len(batch[i]['Electron.PT'])):
            charges_e.append(batch[i]['Electron.Charge'][e])

        for m in range(len(batch[i]['Muon.PT'])):
            charges_m.append(batch[i]['Muon.Charge'][m])

        #only allow same flavour if same charge
        if len(charges_e) == 2:
            if charges_e[0] == charges_e[1]:
                n_sameQ += 1
        if len(charges_m) == 2:
            if charges_m[0] == charges_m[1]:
                n_sameQ += 1

        if nElectrons + nMuons == 3 and nJets < 5 and n_sameQ > 0:
            charges = charges_m + charges_e
            if charges[0] == charges[1] == charges[2]:
                pass_selection[i] = False
            else:
                pass_selection[i] = True

    return pass_selection


# selects which data type we are putting through the processing
def batch_selector(batch, decay):
    higgs = checkHiggs(batch)
    presel = preselection(batch)
    if decay == 0:  # VH
        batch = batch[np.logical_and(higgs, presel)]
    elif decay == 1:  # WWW
        batch = batch[np.logical_and(~higgs, presel)]
    else:  # other ie bkg
        batch = batch[presel]

    return batch

@numba.jit(nopython=True)
def get_leptons(batch):
    # get l0, l1, l2 where l0 has unique charge, l1 is closest to it and l2 is the other
    # use these leptons to compute properties
    # return properties to be used in ML network
    PT_l0 = np.full((len(batch),), 1000.)
    PT_l1 = np.full((len(batch),), 1000.)
    PT_l2 = np.full((len(batch),), 1000.)
    invarM_l0l2 = np.full((len(batch),), 1000.)
    invarM_l1l2 = np.full((len(batch),), 1000.)
    invarM_l0l1 = np.full((len(batch),), 1000.)
    E_l1l2 = np.full((len(batch),), 1000.)
    E_l0l2 = np.full((len(batch),), 1000.)
    E_l0l1 = np.full((len(batch),), 1000.)
    transM_l1l2 = np.full((len(batch),), 1000.)
    transM_l0l2 = np.full((len(batch),), 1000.)
    transM_l0l1 = np.full((len(batch),), 1000.)
    deltaeta_l1l2 = np.full((len(batch),), 1000.)
    deltaeta_l0l2 = np.full((len(batch),), 1000.)
    deltaeta_l0l1 = np.full((len(batch),), 1000.)
    dR_l1l2 = np.full((len(batch),), 1000.)
    dR_l0l1 = np.full((len(batch),), 1000.)
    dR_l0l2 = np.full((len(batch),), 1000.)
    totalP = np.full((len(batch),), 1000.)
    DRll = np.full((len(batch),), 1000.)
    invarM = np.full((len(batch),), 1000.)
    n_jet = np.full((len(batch),), 1000.)
    n_btag = np.full((len(batch),), 1000.)
    max_PT_jet = np.full((len(batch),), 1000.)
    d0_l0 = np.full((len(batch),), 1000.)
    d0_l1 = np.full((len(batch),), 1000.)
    d0_l2 = np.full((len(batch),), 1000.)
    zsin_l0 = np.full((len(batch),), 1000.)
    zsin_l1 = np.full((len(batch),), 1000.)
    zsin_l2 = np.full((len(batch),), 1000.)
    theta_l0 = np.full((len(batch),), 1000.)
    theta_l1 = np.full((len(batch),), 1000.)
    theta_l2 = np.full((len(batch),), 1000.)
    dPhi_metl0 = np.full((len(batch),), 1000.)
    dPhi_metl1 = np.full((len(batch),), 1000.)
    dPhi_metl2 = np.full((len(batch),), 1000.)
    f_alpha: ndarray = np.full((len(batch),), 1000.)
    met: ndarray = np.full((len(batch),), 1000.)

    for i in range(len(batch)):
        leptons = []
        lepton0s = []
        charges = []
        d0 = []
        # d0sig = []
        dz = []
        n_jet[i] = 0
        n_btag[i] = 0
        # loop over electrons
        for e in range(len(batch[i]['Electron.PT'])):
            l = vector.obj(pt=batch[i]['Electron.PT'][e],
                           phi=batch[i]['Electron.Phi'][e],
                           eta=batch[i]['Electron.Eta'][e],
                           mass=511. / 1e6)

            charges.append(batch[i]['Electron.Charge'][e])
            d0.append(batch[i]['Electron.D0'][e])
            # d0sig.append(batch[i]['Electron.ErrorD0'][e])
            dz.append(batch[i]['Electron.DZ'][e])
            leptons.append(l)
        #print(leptons, 'after e')

        # loop over muons
        for m in range(len(batch[i]['Muon.PT'])):
            l = vector.obj(pt=batch[i]['Muon.PT'][m],
                           phi=batch[i]['Muon.Phi'][m],
                           eta=batch[i]['Muon.Eta'][m],
                           mass=105.66 / 1e3)

            charges.append(batch[i]['Muon.Charge'][m])
            leptons.append(l)
            d0.append(batch[i]['Muon.D0'][m])
            # d0sig.append(batch[i]['Muon.ErrorD0'][m])
            dz.append(batch[i]['Muon.DZ'][m])
        #print(leptons, 'after m')

        jet_pt = []
        for j in range(len(batch[i]['Jet.PT'])):
            jet_pt.append(batch[i]['Jet.PT'][j])
            n_jet[i] += 1
        for b in range(len(batch[i]['Jet.BTag'])):
            if batch[i]['Jet.BTag'][b] == 1:
                n_btag[i] += 1
            else:
                continue
        if n_jet[i] == 0:
            max_PT_jet[i] = 0
        else:
            max_PT_jet[i] = np.max((np.array(jet_pt)))

        metvec = vector.obj(pt=batch[i]['MissingET.MET'][0],
                       phi=batch[i]['MissingET.Phi'][0],
                       eta=0,
                       mass=0)

        corr_charges = []
        # find lepton 0 (has unique charge)
        if charges[0] == charges[1]:
            corr_charges.append(charges[2])
            lepton0s.append(leptons[2])
        elif charges[0] == charges[2]:
            corr_charges.append(charges[1])
            lepton0s.append(leptons[1])
        else:
            corr_charges.append(charges[0])
            lepton0s.append(leptons[0])
        # TODO add charges of l1 and l2 in order

        # find lepton 1 (closest to l0) and label the remaining one lepton 2
        l0 = lepton0s[0]
        mindr = DRll[i]
        for l in leptons:
            if l == l0:
                continue
            if l.deltaR(l0) < mindr:
                mindr = l.deltaR(l0)
                l1 = l
        for l in leptons:
            if l == l0:
                continue
            elif l == l1:
                continue
            else:
                l2 = l
        DRll[i] = mindr
        idx_l1 = leptons.index(l1)
        idx_l2 = leptons.index(l2)
        corr_charges.append(charges[idx_l1])
        corr_charges.append(charges[idx_l2])

        # categorise the d0, d0sig and dz data for each lepton
        for a in range(0, 3):
            if leptons[a] == l0:
                d0_l0[i] = d0[a]
                # d0sig_l0 = d0sig[a]
                dz_l0 = dz[a]
            elif leptons[a] == l1:
                d0_l1[i] = d0[a]
                # d0sig_l1 = d0sig[a]
                dz_l1 = dz[a]
            elif leptons[a] == l2:
                d0_l2[i] = d0[a]
                # d0sig_l2 = d0sig[a]
                dz_l2 = dz[a]
        # should be d0 / d0sig
        # for now leave as d0s

        # from pseudorapidity eta = -ln(tan(theta/2))
        theta_l0[i] = 2 * np.arctan(np.exp(-l0.eta))
        theta_l1[i] = 2 * np.arctan(np.exp(-l1.eta))
        theta_l2[i] = 2 * np.arctan(np.exp(-l2.eta))
        zsin_l0[i] = dz_l0 * np.sin(theta_l0[i])
        zsin_l1[i] = dz_l1 * np.sin(theta_l1[i])
        zsin_l2[i] = dz_l2 * np.sin(theta_l2[i])

        dPhi_metl0[i] = np.abs(metvec.phi - l0.phi)
        dPhi_metl1[i] = np.abs(metvec.phi - l1.phi)
        dPhi_metl2[i] = np.abs(metvec.phi - l2.phi)

        PT_l0[i] = l0.p
        PT_l1[i] = l1.p
        PT_l2[i] = l2.p

        invarM[i] = (l0 + l1 + l2).M

        """""
        genpar = []
        other= []
        n_tau = 0
        for j in range(len(batch[i]['Particle.Mass'])):
            if abs(batch[i]['Particle.PID'][j]) == 16:
                n_tau += 1
            if n_tau == 2:
                print('Z to tau tau')
                f_alpha[i] = F_alpha(l0, l1, l2, metvec)
            else:
                f_alpha[i] = None

        
            if batch[i]['Particle.Status'][j] == 1:
                if abs(batch[i]['Particle.PID'][j]) :
                    eh = vector.obj(pt=batch[i]['Particle.PT'][j],
                                   phi=batch[i]['Particle.Phi'][j],
                                   eta=batch[i]['Particle.Eta'][j],
                                   mass=batch[i]['Particle.Mass'][j])
                    #print(eh)
        #print(genpar, "invarM for genpar")
        """


        invarM_l0l2[i] = (l0 + l2).M
        invarM_l1l2[i] = (l1 + l2).M
        invarM_l0l1[i] = (l0 + l1).M

        E_l1l2[i] = np.sqrt((l1 + l2).pt2 + invarM_l1l2[i])
        E_l0l2[i] = np.sqrt((l0 + l2).pt2 + invarM_l0l2[i])
        E_l0l1[i] = np.sqrt((l0 + l1).pt2 + invarM_l0l1[i])

        transM_l1l2[i] = np.sqrt(np.abs(((E_l1l2[i] + metvec.pt)** 2 - (np.abs(l1.pt * l2.pt + metvec.pt))** 2)))
        transM_l0l2[i] = np.sqrt(np.abs(((E_l0l2[i] + metvec.pt)** 2 - (np.abs(l0.pt * l2.pt + metvec.pt))** 2)))
        transM_l0l1[i] = np.sqrt(np.abs(((E_l0l1[i] + metvec.pt)** 2 - (np.abs(l0.pt * l1.pt + metvec.pt))** 2)))

        deltaeta_l1l2[i] = l1.deltaeta(l2)
        deltaeta_l0l2[i] = l0.deltaeta(l2)
        deltaeta_l0l1[i] = l0.deltaeta(l1)

        # deltaangle_l1l2[i]=l1.deltaangle(l2)
        # deltaangle_l0l2[i]=l0.deltaangle(l2)
        # deltaangle_l0l1[i]=l0.deltaangle(l1)

        dR_l1l2[i] = l1.deltaR(l2)
        dR_l0l1[i] = l0.deltaR(l1)
        dR_l0l2[i] = l0.deltaR(l2)

        totalP[i] = l0.p + l1.p + l2.p

        f_alpha[i] = F_alpha(l0,l1,l2,metvec,corr_charges)

        met[i] = metvec.pt

    return PT_l0, PT_l1, PT_l2, met, \
           dR_l0l1, dR_l0l2, dR_l1l2, \
           deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
           transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
           invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
           dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
           d0_l0, d0_l1, d0_l2, n_jet, \
           n_btag, zsin_l0, zsin_l1, zsin_l2 , f_alpha
    # no S to compute S/met
    # cannot compute max(1 - E/p)

@numba.jit(nopython=True)
def F_alpha(l0, l1, l2, met, charges):
    #need to relabel to Falpha calc vars ell1, ell2, ell3
    #ell1 = minimum pT lepton
    #identify lepton(s) with opposite charge
    #ell2 = lepton w/opposite charge to ell1 and 2nd lowest pT
    #ell3 = remaining lepton

    lepton_list = [l0, l1, l2]
    pt_list = []
    for i, element in enumerate(lepton_list):
        pt_list.append(element.pt)
    ell1 = lepton_list[pt_list.index(min(pt_list))]

    ell1_charge = charges[pt_list.index(min(pt_list))]

    #list of only opposite charges to ell1
    idx_opQ = []
    for i in range(0, 3):
        if charges[i] == ell1_charge:
            pass
        else:
            idx_opQ.append(i)

    if len(idx_opQ) == 1:
        ell2 = lepton_list[idx_opQ[0]]
        lepton_list.remove(ell1)
        lepton_list.remove(ell2)
        ell3 = lepton_list[0]

    elif len(idx_opQ) == 2:
        pt_1 = pt_list[idx_opQ[0]]
        pt_2 = pt_list[idx_opQ[1]]
        if pt_1 == pt_2: #when pT are the same (rare)
            print("same")
            ell2 = lepton_list[idx_opQ[0]]
            ell3 = lepton_list[idx_opQ[1]]
        else: #min pT lepton is ell2
            if pt_1 < pt_2:
                ell2 = lepton_list[idx_opQ[0]]
                ell3 = lepton_list[idx_opQ[1]]
            else:
                ell2 = lepton_list[idx_opQ[1]]
                ell3 = lepton_list[idx_opQ[0]]

    else:
        print("There should not be non unique charge")
        #ell3 = lepton_list[pt_list.index(max(pt_list))]
        #lepton_list.remove(ell1)
        #lepton_list.remove(ell3)
        #ell2 = lepton_list[0]


    trial = 100
    stepsize = 0.01
    successes = 0
    binweight = [0.000523209,0.0185422,0.0773298,0.1551565,0.19408925,0.1874,0.15113275,0.12033775,0.0721813,0.02330715]

    a1 = 91.1876**2
    b1 = 1.77682**2
    e1 = ell2.E ** 2
    a2 = (80.385**2)/2
    b2 = ell3.E
    e2 = -(ell3.pz)
    for i in range(1, trial):
        alpha1 = 1 / (i*stepsize)
        c1 = alpha1 * (ell1.E) * (ell2.E)
        if ((alpha1 **2) * (ell1.E) * (ell1.E - b1))>=0:
            d1 = np.sqrt(alpha1**2 * ell1.E * (ell1.E - b1) * np.cos(ell1.deltaangle(ell2)))
        else:
            continue

        f1 = a1*a1*d1*d1*e1 - 4*a1*b1*d1*d1*e1 + 4*b1*b1*d1*d1*e1 - 4*b1*c1*c1*d1*d1 + 4*b1*d1*d1*d1*d1*e1
        g1 = c1**2 - (d1**2)*e1

        if np.logical_and((f1>=0),(g1!=0)):
            alpha2_1 = (np.sqrt(f1) + a1*c1 - 2*b1*c1) / (2*g1)
            alpha2_2 = (-np.sqrt(f1) + a1*c1 - 2*b1*c1) / (2*g1)
        else:
            continue

        if np.logical_and((alpha2_1<1), (alpha2_2<1)):
            continue

        if alpha2_1 > 1:
            nuPx_1 = met.x - ((alpha1-1)*(ell1.E * np.sin(ell1.theta))) * np.cos(ell1.phi) - ((alpha2_1-1)*(ell2.E * np.sin(ell2.theta) * np.cos(ell2.phi)))
            nuPy_1 = met.y - ((alpha1 - 1) * (ell1.E * np.sin(ell1.theta))) * np.sin(ell1.phi) - (
                        (alpha2_1 - 1) * (ell2.E * np.sin(ell2.theta) * np.sin(ell2.phi)))

            c2_1 = nuPx_1**2 +nuPy_1**2
            d2_1 = - (ell3.px*nuPx_1) - (ell3.py*nuPy_1)

            f2_1 = a2**2*b2**2 - 2*a2*b2**2*d2_1 - b2**4*c2_1 + b2**2*c2_1*e2**2 + b2**2*d2_1**2
            g2_1 =  b2**2 - e2**2

            if np.logical_and(f2_1>=0, g2_1!=0):
                if int((i+1)/10) < 10:
                    successes += binweight[int((i+1)/10)]
                else:
                    successes += binweight[int((i)/10)]

        if alpha2_2 > 1:
            nuPx_2 = met.x - ((alpha1-1)*(ell1.E * np.sin(ell1.theta))) * np.cos(ell1.phi) - ((alpha2_2-1)*(ell2.E * np.sin(ell2.theta) * np.cos(ell2.phi)))
            nuPy_2 = met.y - ((alpha1-1)*(ell1.E * np.sin(ell1.theta))) * np.sin(ell1.phi) - ((alpha2_2-1)*(ell2.E * np.sin(ell2.theta) * np.sin(ell2.phi)))

            c2_2 = nuPx_2**2 + nuPy_2**2
            d2_2 = - (ell3.px*nuPx_2) - (ell3.py*nuPy_2)

            f2_2 = a2**2*b2**2 - 2*a2*b2**2*d2_2 -  b2**4*c2_2 +  b2**2*c2_2*e2**2 + b2**2*d2_2**2
            g2_2 = b2**2 - e2**2

            if np.logical_and(f2_2>=0, g2_2!=0):
                if int((i+1)/10) < 10:
                    successes += binweight[int((i+1)/10)]
                else:
                    successes += binweight[int((i)/10)]

    corrector = 0
    for j in range(0, 10):
        corrector = corrector + 10*binweight[j]

    retval = successes / (2*corrector)
    return retval

def plothist(df, featurelist, binend, numberbins, df2=None, weight2=None, name2=None):
    #plots distributions for input data
    #if df2 is not None, can be used to plot EFT inputs, or output distributions post ML
    weight_sig, weight_bkg, lumi = get_weights()

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

        plt.show()

def df_selection(df):
    selection_pt = (df["PT_l0"] > 15) & (df["PT_l1"] > 15) & (df["PT_l2"] > 15)
    df = df[selection_pt]

    selection_mindelR = (df["delR_l0l1"] > 0.1)
    df = df[selection_mindelR]

    return df

def n_btag_selec(df):
    selection_nbtag = (df["n_btag"] == 0)
    df = df[selection_nbtag]
    return df


@numba.jit(nopython=True)
def node_outputs(y_test, y_pred):
    VH_dist_truthVH = []
    VH_dist_truthWWW = []
    VH_dist_truthbkg = []
    for i in range(0, len(y_pred)):
        if y_test[i][0] == 1:
            VH_dist_truthVH.append(y_pred[i][0])
        elif y_test[i][1] == 1:
            VH_dist_truthWWW.append(y_pred[i][0])
        else:
            VH_dist_truthbkg.append(y_pred[i][0])
    WWW_dist_truthVH = []
    WWW_dist_truthWWW = []
    WWW_dist_truthbkg = []
    for i in range(0, len(y_pred)):
        if y_test[i][0] == 1:
            WWW_dist_truthVH.append(y_pred[i][1])
        elif y_test[i][1] == 1:
            WWW_dist_truthWWW.append(y_pred[i][1])
        else:
            WWW_dist_truthbkg.append(y_pred[i][1])
    bkg_dist_truthVH = []
    bkg_dist_truthWWW = []
    bkg_dist_truthbkg = []
    for i in range(0, len(y_pred)):
        if y_test[i][0] == 1:
            bkg_dist_truthVH.append(y_pred[i][2])
        elif y_test[i][1] == 1:
            bkg_dist_truthWWW.append(y_pred[i][2])
        else:
            bkg_dist_truthbkg.append(y_pred[i][2])

    lists = [[VH_dist_truthVH, VH_dist_truthWWW, VH_dist_truthbkg],
             [WWW_dist_truthVH, WWW_dist_truthWWW, WWW_dist_truthbkg],
             [bkg_dist_truthVH, bkg_dist_truthWWW, bkg_dist_truthbkg]]

    return lists
def plot_nodes(y_test, y_pred, y_test_EFT=None, y_pred_EFT=None, weight_EFT=None, name=None):
    listsSM = node_outputs(y_test, y_pred)

    names = ["VH", "WWW", 'bkg']
    weight_sig, weight_bkg, lumi = get_weights()  # weights & luminosity

    if y_test_EFT is not None:
            listsEFT = node_outputs(y_test_EFT, y_pred_EFT)

    for i in range(0, 3):
        listSM = listsSM[i]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bins = np.linspace(0., 1., 40)
        ax.hist(listSM, color=['purple', 'lime', 'cyan'], bins=bins,
                weights=[np.full(len(listSM[0]), weight_sig),
                         np.full(len(listSM[1]), weight_sig),
                         np.full(len(listSM[2]), weight_bkg)],
                histtype='step', linewidth=5, label=["VH", "WWW", "bkg"])
        if y_test_EFT is not None:
            listEFT = listsEFT[i]
            ax.hist(listEFT, color=['red', 'yellow', 'blue'], bins=bins,
                    weights=[np.full(len(listEFT[0]), weight_EFT),
                             np.full(len(listEFT[1]), weight_EFT),
                             np.full(len(listEFT[2]), weight_bkg)],
                    histtype='step', linewidth=5, label=["VH", "WWW", "bkg"])
            ax.set_xlabel(f"Network Output Distribution for {names[i]} for EFT {name}")
        else:
            ax.set_xlabel(f"Network Output Distribution for {names[i]}")

        ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
        ax.set_yscale("log")
        ax.set_ylabel("Entries")
        ax.legend()
        if y_test_EFT is not None:
            plt.savefig(
                f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/{names[i]}_EFT_{name}_node.png")
        else:
            plt.savefig(
                f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/{names[i]}_node.png")
        plt.show()

@numba.jit(nopython=True)
def EFT_s_over_b(y_test, y_pred, weightEFT, weight_sig=get_weights()[0]):
    SM_dist_truthSM, \
        SM_dist_truthEFT, \
        EFT_dist_truthSM, \
        EFT_dist_truthEFT = get_EFT_nodes(y_test, y_pred)

    EFT_dist_truthEFT.sort() #put in ascending order
    EFT_dist_truthSM.sort()

    #we compute amount of EFT events at EFT node versus amount of SM events
    #SM gets the type 1, EFT gets type 0
    #the EFTs should be classified near 0
    optlist = []
    for i in range(0, len(EFT_dist_truthEFT)):
        optlist.append([EFT_dist_truthEFT[i], weightEFT])


    #optlist = optlist[::-1]  # reverse the lists - now descending order
    #bkglist = bkglist[::-1]

    n_sig = 4 / weightEFT  # number of signals to get to 4 events

    optlist = optlist[0:int(n_sig)]  # cut signal at 4 events

    #we are using EFT truth SM as the background
    signals = [i[0] for i in optlist]
    value = min(signals)  # cutoff value
    EFT_dist_truthSM[:] = [x for x in EFT_dist_truthEFT if x <= value]  # slice bkg at cutoff

    if len(EFT_dist_truthSM) < 1:
        return 1  # common error if no items in bkglist
    else:
        index = EFT_dist_truthSM.index(min(EFT_dist_truthSM))  # location of item at cutoff

        n_bkg = index * weight_sig #number bkg events when 4 signal events

        if n_bkg > 0:
            s_over_bkg_EFT = n_sig / n_bkg
        else:
            s_over_bkg_EFT = 1  # when no bkg

    return s_over_bkg_EFT

@numba.jit(nopython=True)
def get_EFT_nodes(y_test, y_pred):
    SM_dist_truthSM = []
    SM_dist_truthEFT = []
    for i in range(0, len(y_pred)):
        if y_test[i] == 1:
            SM_dist_truthSM.append(y_pred[i])
        elif y_test[i] == 0:
            SM_dist_truthEFT.append(y_pred[i])
    EFT_dist_truthEFT = []
    EFT_dist_truthSM = []
    for i in range(0, len(y_pred)):
        if y_test[i] == 1:
            EFT_dist_truthSM.append(y_pred[i])
        elif y_test[i] == 0:
            EFT_dist_truthEFT.append(y_pred[i])

    return SM_dist_truthSM, SM_dist_truthEFT, \
        EFT_dist_truthSM, EFT_dist_truthEFT

def plot_nodes_2(y_test, y_pred, weightEFT):
    SM_dist_truthSM, \
        SM_dist_truthEFT, \
        EFT_dist_truthSM, \
        EFT_dist_truthEFT = get_EFT_nodes(y_test, y_pred)

    weight_sig, weight_bkg, lumi = get_weights()  # weights & luminosity

    names = ["SM", "EFT"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(0, 1, 40)
    ax.hist([SM_dist_truthSM, SM_dist_truthEFT], color=['purple', 'cyan'], bins=bins,
            weights=[np.full(len(SM_dist_truthSM), weight_sig),
                     np.full(len(SM_dist_truthEFT), weightEFT)],
            histtype='step', linewidth=5, label=["truth SM", "truth EFT"])
    ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
    ax.set_xlabel(f"Network Output Distribution for SM Node")
    #ax.set_yscale("log")
    ax.set_ylabel("Entries")
    ax.legend()
    plt.savefig(
        f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/SM_node.png")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(0, 1, 40)
    ax.hist([EFT_dist_truthSM, EFT_dist_truthEFT], color=['purple', 'lime'], bins=bins,
            weights=[np.full(len(EFT_dist_truthSM), weight_sig),
                     np.full(len(EFT_dist_truthEFT), weightEFT)],
            histtype='step', linewidth=5, label=["truth SM", "truth EFT"])
    ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
    ax.set_xlabel(f"Network Output Distribution for EFT Node")
    ax.set_yscale("log")
    ax.set_ylabel("Entries")
    ax.legend()
    plt.savefig(
        f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/EFT_node.png")
    plt.show()

def sigoverbkg(type, y_pred, y_test, weight=get_weights()[0], name_trial=None, y_pred_EFT=None,
               y_test_EFT=None, weightEFT=None, nameEFT=None, graph=False):
    # function to determine S/B ratios
    # can plot node outputs for SM data AND EFT data on top
    # be careful with inputs

    if type == 'VH': # VH is 1st column
        position = int(0)
        color1 = 'purple'
        color2 = 'red'
    elif type == 'WWW': # WWW is 2nd column
        position = int(1)
        color1 = 'lime'
        color2 = 'yellow'

    # generate lists for what has been identified as signal
    # X = signal type ie VH or WWW
    # X_dist_truthX = signal identified as signal
    # X_dist_truth_bkg = bkg identified as signal
    X_dist_truthX = []
    X_dist_truthbkg = []
    for i in range(0, len(y_pred)):
        if y_test[i][position] == 1:
            X_dist_truthX.append(y_pred[i][position])
        elif y_test[i][2] == 1:
            X_dist_truthbkg.append(y_pred[i][2])
        else:
            pass
    X_dist_truthX.sort() #put in ascending order
    X_dist_truthbkg.sort()

    weight_sig, weight_bkg, lumi = get_weights()  # weights & luminosity

    optlist = []
    for i in range(0,len(X_dist_truthX)):
        optlist.append([X_dist_truthX[i], weight_sig])

    bkglist = []
    for i in range(0,len(X_dist_truthbkg)):
        bkglist.append([X_dist_truthbkg[i], weight_bkg])

    optlist = optlist[::-1] # reverse the lists - now descending order
    bkglist = bkglist[::-1]

    n_sig = 4 / weight # number of signals to get to 4 events

    optlist = optlist[0:int(n_sig)] #cut signal at 4 events
    signals = [i[0] for i in optlist]
    value = min(signals) #cutoff value
    bkglist = [i[0] for i in bkglist]
    bkglist[:] = [x for x in bkglist if x >= value] #slice bkg at cutoff

    if y_test_EFT is not None: #if also evaluating EFT data
        X_dist_truthX_EFT = []
        X_dist_truthbkg_EFT = []
        for i in range(0, len(y_pred_EFT)):
            if y_test_EFT[i][position] == 1:
                X_dist_truthX_EFT.append(y_pred_EFT[i][position])
            elif y_test_EFT[i][2] == 1:
                X_dist_truthbkg_EFT.append(y_pred_EFT[i][2])
            else:
                pass
        X_dist_truthX_EFT.sort()
        X_dist_truthbkg_EFT.sort()

        X_dist_truthX_EFT[:] = [x for x in X_dist_truthX_EFT if x >= value]  # slice at cutoff
        X_dist_truthbkg_EFT[:] = [x for x in X_dist_truthbkg_EFT if x >= value]  # slice at cutoff

    # TODO make graphs only print if variable in input set to True, default False
    if graph is not False:    # use this to check cutoff is correct
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bins = np.linspace(0.95, 1, 40)
        ax.hist([signals, bkglist], color=[color1, 'cyan'], bins=bins,
                weights=[np.full(len(signals), weight_sig),
                         np.full(len(bkglist), weight_bkg)],
                histtype='step', linewidth=5, label=[f"{type}", "bkg"])
        if y_pred_EFT is not None:
            ax.hist([X_dist_truthX_EFT, X_dist_truthbkg_EFT], color=[color2, 'blue'], bins=bins,
                    weights=[np.full(len(X_dist_truthX_EFT), weightEFT),
                             np.full(len(X_dist_truthbkg_EFT), weight_bkg)],
                    histtype='step', linewidth=5, label=[f"{type} {nameEFT}", f"bkg {nameEFT}"])
            ax.set_xlabel(f"Network Output Distribution for {type} for EFT {nameEFT}")
        elif nameEFT is not None:
            ax.set_xlabel(f"Network Output Distribution for {type} for EFT {nameEFT}")
        else:
            ax.set_xlabel(f"Network Output Distribution for {type}")

        ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
        ax.set_yscale("log")
        ax.set_ylabel("Entries")
        ax.legend()

        #save fig for EFT node distributions
        if y_pred_EFT is not None:
            plt.savefig(
                f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/cutoff_{nameEFT}_node_{type}.png")

        plt.show()

    if len(bkglist) < 1:
        return 1 # common error if no items in bkglist
    else:
        index = bkglist.index(min(bkglist))  # location of item at cutoff

        n_bkg = index * weight_bkg

        if n_bkg > 0:
            s_over_bkg = n_sig / n_bkg
        else:
            s_over_bkg = 1 # when no bkg

        #print(s_over_bkg, f'{type} signal over bkg')

        return s_over_bkg #returns the value
def pre_processing(root_signal, root_bkg, name):
    # function to produce a dataset with all necessary cuts and preselection
    # returned dataset has Type column for VH, WWW or bkg

    N_root_sig = getNeventsRoot(root_signal)
    N_root_bgk = 2964674 #saves computing it

    ntotal_VH = 0
    nselected_VH = 0
    df_VH = None

    myfilter = "/(MissingET.(MET|Phi)|Particle.(PID|Status)|Electron.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Muon.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Jet.(PT|Eta|Phi|BTag))/"
    batch_size = 5000 #dataset too large to process all at once

    iterator = uproot.iterate([f + ":Delphes" for f in root_signal], \
                              step_size=batch_size, filter_name=myfilter)  # iterator for the 2 signals

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
            n_btag, zsin_l0, zsin_l1, zsin_l2, f_alpha = get_leptons(batch)
        # cannot compute max(1 - E/p)

        dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
                   'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
                   'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
                   'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2,
                   'z0sintheta_l0': zsin_l0, 'z0sintheta_l1': zsin_l1, 'z0sintheta_l2': zsin_l2,
                   'n_btag': n_btag, 'max_PT_jet': max_PT_jet,
                   'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
                   'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2': invarM_l1l2,
                   'm_lll': invarM, 'F_alpha': f_alpha}
        # make into dict

        if df_VH is not None:
            newdata_VH = pd.DataFrame(dataset)
            df_VH = pd.concat([df_VH, newdata_VH], axis=0, ignore_index=True)
            #concatenates new data with previous batch's data
        else:
            df_VH = pd.DataFrame(dataset)
            #for the first round where there is no data

    print("Preselection efficiency for VH events: %2.2f %%" % (100. * nselected_VH / ntotal_VH))
    df_VH.insert(loc=0, column='Type', value=0) # VH is event type 0
    df_VH = n_btag_selec(df_VH) #no B tagged jets in VH

    myfilter = "/(MissingET.(MET|Phi)|Particle.(PID|Status)|Electron.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Muon.(PT|Eta|Phi|Charge|D0|ErrorD0|DZ)|Jet.(PT|Eta|Phi|BTag))/"
    batch_size = 5000

    iterator = uproot.iterate([f + ":Delphes" for f in root_signal], \
                              step_size=batch_size, filter_name=myfilter)  # iterator for the 2 signals

    ntotal_WWW = 0
    nselected_WWW = 0
    df_WWW = None

    for batch in tqdm(iterator, total=N_root_sig // batch_size):
        higgs = checkHiggs(batch)
        presel = preselection(batch)
        ntotal_WWW += (~higgs).sum() #NOT Higgs in WWW signal
        nselected_WWW += np.logical_and(~higgs, presel).sum()

        batch = batch_selector(batch, 1)  # WWW data

        PT_l0, PT_l1, PT_l2, met, \
            dR_l0l1, dR_l0l2, dR_l1l2, \
            deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
            transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
            invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
            dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
            d0_l0, d0_l1, d0_l2, n_jet, \
            n_btag, zsin_l0, zsin_l1, zsin_l2, f_alpha = get_leptons(batch)
        # cannot compute max(1 - E/p)

        dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
                   'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
                   'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
                   'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2,
                   'z0sintheta_l0': zsin_l0, 'z0sintheta_l1': zsin_l1, 'z0sintheta_l2': zsin_l2,
                   'n_btag': n_btag, 'max_PT_jet': max_PT_jet,
                   'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
                   'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2': invarM_l1l2,
                   'm_lll': invarM, 'F_alpha': f_alpha}

        if df_WWW is not None:
            newdata_WWW = pd.DataFrame(dataset)
            df_WWW = pd.concat([df_WWW, newdata_WWW], axis=0, ignore_index=True)
            # concatenates new data with previous batch's data
        else:
            df_WWW = pd.DataFrame(dataset)
            # for the first round where there is no data

    print("Preselection efficiency for WWW events: %2.2f %%" % (100. * nselected_WWW / ntotal_WWW))

    df_WWW.insert(loc=0, column='Type', value=1) # WWW is event type 1
    df_WWW = n_btag_selec(df_WWW) #no B tagged jets for WWW

    # loop over background
    ntotal = 0
    nselected = 0
    df_bkg = None
    batch_size = 5000
    iterator = uproot.iterate([f + ":Delphes" for f in root_bkg], \
                              step_size=batch_size, filter_name=myfilter)  # different as uses root_bkg file
    for batch in tqdm(iterator, total=N_root_bgk / batch_size):
        higgs = checkHiggs(batch)
        presel = preselection(batch) # still have to apply preselection:
                                    # we won't know which data types we are
                                    # looking at when using real data

        batch = batch_selector(batch, None)
        PT_l0, PT_l1, PT_l2, met, \
            dR_l0l1, dR_l0l2, dR_l1l2, \
            deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
            transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
            invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
            dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
            d0_l0, d0_l1, d0_l2, n_jet, \
            n_btag, zsin_l0, zsin_l1, zsin_l2, f_alpha = get_leptons(batch)
        # cannot compute max(1 - E/p)

        dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
                   'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
                   'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
                   'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2,
                   'z0sintheta_l0': zsin_l0, 'z0sintheta_l1': zsin_l1, 'z0sintheta_l2': zsin_l2,
                   'n_btag': n_btag, 'max_PT_jet': max_PT_jet,
                   'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
                   'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2': invarM_l1l2,
                   'm_lll': invarM, 'F_alpha': f_alpha}
        # make into dict

        # any round but first round - concatenate with previous batches
        if df_bkg is not None:
            newdata_bkg = pd.DataFrame(dataset)
            df_bkg = pd.concat([df_bkg, newdata_bkg], axis=0, ignore_index=True)

        # first round
        else:
            df_bkg = pd.DataFrame(dataset)

        ntotal += len(batch)
        nselected += presel.sum()
    print("Preselection efficiency for background events: %2.2f %%" % (100. * nselected / ntotal))
    df_bkg.insert(loc=0, column='Type', value=2) #bkg is event type 2

    df_p = pd.concat([df_WWW, df_VH, df_bkg], axis=0, ignore_index=True)
    df_p = df_selection(df_p)
    df_p.to_csv(f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_{name}.csv",
                index=False)

def plot_modelhistory(model_history, metric):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(np.sqrt(model_history.history[metric]), 'r', label='Training Data')
    ax.plot(np.sqrt(model_history.history[f'val_{metric}']), 'b', label='Validation Data')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(f'{metric}', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.savefig(f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/{metric}_with_epoch.png")
    plt.show()


def get_VH_WWW_df(model, name, filerange, df_test, y_pred, y_array):
    df_EFT = pd.read_csv(
        f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_{name}.csv")
    weight_sig_EFT = get_weights_EFT(name, filerange)

    x_EFT = df_EFT.drop('Type', axis=1)
    sc = RobustScaler()
    x_EFT = pd.DataFrame(sc.fit_transform(x_EFT))

    Y_EFT = df_EFT['Type'].values

    y_cat_EFT = to_categorical(Y_EFT, num_classes=3)

    X_EFT = torch.tensor(x_EFT.values, dtype=torch.float32)
    y_EFT = torch.tensor(y_cat_EFT, dtype=torch.float32).reshape(-1, 3)

    # Get df post-training using the classifications predicted by the model 'y_test_class'
    y_pred_EFT = model(X_EFT)
    y_array_EFT = y_EFT.detach().numpy()
    y_pred_EFT = y_pred_EFT.detach().numpy()
    sigoverbkg('VH', y_pred_EFT, y_array_EFT, nameEFT=name, graph=True)
    sigoverbkg('WWW', y_pred_EFT, y_array_EFT, nameEFT=name, graph=True)

    y_pred_class_EFT = np.argmax(y_pred_EFT, axis=1)  # classes predicted for training data
    y_truth_class_EFT = np.argmax(y_array_EFT, axis=1)  # classes predicted for test data using model

    print(classification_report(y_truth_class_EFT, y_pred_class_EFT))

    featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met',
                    'delR_l0l1', 'delR_l0l2', 'delR_l1l2',
                    'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2',
                    'dPhi_MET_l0', 'dPhi_MET_l1', 'dPhi_MET_l2',
                    'z0sintheta_l0', 'z0sintheta_l1', 'z0sintheta_l2',
                    'n_btag', 'max_PT_jet',
                    'mT_l0l1', 'mT_l0l2', 'mT_l1l2', 'sumPT',
                    'm_l0l1', 'm_l0l2', 'm_l1l2',
                    'm_lll', 'F_alpha']

    x_EFT = np.array(sc.inverse_transform(x_EFT))
    df_EFT_test = pd.DataFrame(x_EFT, columns=featurenames)
    df_EFT_test.insert(loc=0, column='Type', value=y_pred_class_EFT)

    sigoverbkg('VH', y_pred, y_array, y_pred_EFT=y_pred_EFT, y_test_EFT=y_EFT,
               weightEFT=weight_sig_EFT, nameEFT=name)
    sigoverbkg('WWW', y_pred, y_array, y_pred_EFT=y_pred_EFT, y_test_EFT=y_EFT,
               weightEFT=weight_sig_EFT, nameEFT=name)

    # histograms to compare input and output hists
    plothist(df_EFT, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_EFT_test, weight2=weight_sig_EFT,
             name2='Post-ML')
    plothist(df_EFT, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_EFT_test, weight2=weight_sig_EFT,
             name2='Post-ML')
    plothist(df_EFT, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_EFT_test, weight2=weight_sig_EFT,
             name2='Post-ML')
    plothist(df_EFT, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4., 30, df2=df_EFT_test,
             weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["m_lll"], 1500., 50, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["sumPT"], 1500., 100, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["n_btag"], 10., 9, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["met"], 500., 100, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["max_PT_jet"], 500., 100, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')
    plothist(df_EFT, ["F_alpha"], 1., 20, df2=df_EFT_test, weight2=weight_sig_EFT, name2='Post-ML')

    df_test.insert(loc=0, column="SM", value=1)
    df_EFT_test.insert(loc=0, column="SM", value=0)

    df_EFT_NN = pd.concat([df_test, df_EFT_test], axis=0, ignore_index=True)

    cut = (df_EFT_NN["Type"] == 0) | (df_EFT_NN["Type"] == 1)
    df_EFT_NN = df_EFT_NN[cut]

    return df_EFT_NN
