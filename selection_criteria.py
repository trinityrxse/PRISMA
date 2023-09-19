'''

Making all necessary cuts on data

'''

import uproot, pylhe, glob, os
import numpy as np
import numba,vector

def get_weights(filepath):
    lhe_signal = glob.glob(
        f"{filepath}/Samples/Signal/SM/WWW01j_000*.lhe.gz")
    root_signal = glob.glob(
        f"{filepath}/Samples/Signal/SM/WWW01j_000*.root")

    lhe_bkg = glob.glob(f"{filepath}/Samples/3l01j/3l01j_00*.lhe.gz")
    root_bkg = glob.glob(f"{filepath}/Samples/3l01j/3l01j_00*.root")

    xSection_sig = get_xSection(lhe_signal)
    xSection_bkg = get_xSection(lhe_bkg)

    N_lhe_sig = 2000000 #getNeventsLHE(lhe_signal)
    #print(N_lhe_sig)
    N_lhe_bkg = 8000000 #getNeventsLHE(lhe_bkg)
    #print(N_lhe_bkg)

    N_root_sig = 1023602 #getNeventsRoot(root_signal)
    #print(N_root_sig)
    N_root_bgk = 5930312 #getNeventsRoot(root_bkg)
    #print(N_root_bgk)

    # calculate resulting xSection*efficiency
    xSection_sig *= N_root_sig / N_lhe_sig
    xSection_bkg *= N_root_bgk / N_lhe_bkg

    # scale to given luminosity
    lumi = 400  # inverse fb

    n_sig = lumi * xSection_sig * 1000
    n_bkg = lumi * xSection_bkg * 1000

    weight_sig = (n_sig / N_root_sig) / 10 # this is because of weightings in simulated data being wrong
    weight_bkg = (n_bkg / N_root_bgk) / 10 # with other data, do not have /10

    return weight_sig, weight_bkg, lumi

def get_weights_EFT(filepath, EFTname):
    lhe_signal = glob.glob(
        f"{filepath}/Samples/Signal/sig_{EFTname}/WWW01j_00*.lhe.gz")
    root_signal = glob.glob(
        f"{filepath}/Samples/Signal/sig_{EFTname}/WWW01j_00*.root")

    xSection_sig = get_xSection(lhe_signal)

    N_lhe_sig = getNeventsLHE(lhe_signal)

    N_root_sig = getNeventsRoot(root_signal)

    # calculate resulting xSection*efficiency
    xSection_sig *= N_root_sig / N_lhe_sig

    # scale to given luminosity
    lumi = 400  # inverse fb

    n_sig = lumi * xSection_sig * 1000

    weight_sig = (n_sig / N_root_sig) / 10  / 10 # this is because of weightings in simulated data being wrong
                                                # with other data, do not have /10

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
        #if nElectrons + nMuons == 3 and nJets < 5:
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
            leptons.append(l)

        # loop over muons
        for m in range(len(batch[i]['Muon.PT'])):
            l = vector.obj(pt=batch[i]['Muon.PT'][m],
                           phi=batch[i]['Muon.Phi'][m],
                           eta=batch[i]['Muon.Eta'][m],
                           mass=105.66 / 1e3)

            charges.append(batch[i]['Muon.Charge'][m])
            leptons.append(l)
            d0.append(batch[i]['Muon.D0'][m])


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

        dPhi_metl0[i] = np.abs(metvec.phi - l0.phi)
        dPhi_metl1[i] = np.abs(metvec.phi - l1.phi)
        dPhi_metl2[i] = np.abs(metvec.phi - l2.phi)

        PT_l0[i] = l0.p
        PT_l1[i] = l1.p
        PT_l2[i] = l2.p

        invarM[i] = (l0 + l1 + l2).M

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
           f_alpha
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


    trial = 100
    stepsize = 0.01
    successes = 0
    binweight = [0.000523209,0.0185422,0.0773298,0.1551565,\
                 0.19408925,0.1874,0.15113275,0.12033775,0.0721813,0.02330715]

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



def df_selection(df):
    selection_pt = (df["PT_l0"] > 15) & (df["PT_l1"] > 15) & (df["PT_l2"] > 15)
    df = df[selection_pt]

    selection_mindelR = (df["delR_l0l1"] > 0.1)
    df = df[selection_mindelR]

    return df
