import uproot, pylhe, glob, os
import numpy as np
import numba,vector

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

        for e in range(len(batch[i]['Electron.PT'])):
            if batch[i]['Electron.PT'][e] >= 10.0 and abs(batch[i]['Electron.Eta'][e]) < 2.5:
                nElectrons += 1

        for m in range(len(batch[i]['Muon.PT'])):
            if batch[i]['Muon.PT'][m] >= 10.0 and abs(batch[i]['Muon.Eta'][m]) < 2.5:
                nMuons += 1

        for j in range(len(batch[i]['Jet.PT'])):
            if batch[i]['Jet.PT'][j] >= 25.0 and abs(batch[i]['Jet.Eta'][j]) < 2.5:
                nJets += 1

        if nElectrons + nMuons == 3 and nJets < 5:
            pass_selection[i] = True

    return pass_selection

@numba.jit(nopython=True)
def get_MET(batch):
    met=np.zeros((len(batch),))
    metPhi=np.zeros((len(batch),))
    for i in range(len(batch)):
        met[i]=float(batch[i]['MissingET.MET'][0])
        metPhi[i]=float(batch[i]['MissingET.Phi'][0])
    return met, metPhi


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


# get l0, l1, l2 where l0 has unique charge, l1 is closest to it and l2 is the other
@numba.jit(nopython=True)
def get_leptons(batch):
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
    f_alpha = np.full((len(batch),), 1000.)

    met, metPhi = get_MET(batch)
    lepton0s = []
    leptons = []
    for i in range(len(batch)):
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

        jet_pt = []
        for j in range(len(batch[i]['Jet.PT'])):
            jet_pt.append(batch[i]['Jet.PT'][j])
            n_jet[i] += 1
        for b in range(len(batch[i]['Jet.BTag'])):
            n_btag[i] += 1
        if n_jet[i] == 0:
            max_PT_jet[i] = 0
        else:
            max_PT_jet[i] = np.max((np.array(jet_pt)))

        # find lepton 0 (has unique charge)
        if charges[0] == charges[1]:
            lepton0s.append(leptons[2])
        elif charges[0] == charges[2]:
            lepton0s.append(leptons[1])
        else:
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

        dPhi_metl0[i] = np.abs(metPhi[i] - l0.phi)
        dPhi_metl1[i] = np.abs(metPhi[i] - l1.phi)
        dPhi_metl2[i] = np.abs(metPhi[i] - l2.phi)

        PT_l0[i] = l0.p
        PT_l1[i] = l1.p
        PT_l2[i] = l2.p

        invarM[i] = (l0 + l1 + l2).M

        invarM_l0l2[i] = (l0 + l2).M
        invarM_l1l2[i] = (l1 + l2).M
        invarM_l0l1[i] = (l0 + l1).M

        E_l1l2[i] = np.sqrt((l1 + l2).p2 + invarM_l1l2[i])
        E_l0l2[i] = np.sqrt((l0 + l2).p2 + invarM_l0l2[i])
        E_l0l1[i] = np.sqrt((l0 + l1).p2 + invarM_l0l1[i])

        transM_l1l2[i] = np.sqrt(np.abs(((E_l1l2[i] + met[i]) ** 2 - np.abs(l1.p * l2.p + met[i]) ** 2)))
        transM_l0l2[i] = np.sqrt(np.abs(((E_l0l2[i] + met[i]) ** 2 - np.abs(l0.p * l2.p + met[i]) ** 2)))
        transM_l0l1[i] = np.sqrt(np.abs(((E_l0l1[i] + met[i]) ** 2 - np.abs(l0.p * l1.p + met[i]) ** 2)))

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

        f_alpha[i] = F_alpha(l0,l1,l2,met[i])
        print(f_alpha[i])

        # TO DO: F_alpha values from C++ code

    return PT_l0, PT_l1, PT_l2, met, \
           dR_l0l1, dR_l0l2, dR_l1l2, \
           deltaeta_l0l1, deltaeta_l0l2, deltaeta_l1l2, \
           transM_l0l1, transM_l0l2, transM_l1l2, totalP, \
           invarM_l0l1, invarM_l0l2, invarM_l1l2, invarM, \
           dPhi_metl0, dPhi_metl1, dPhi_metl2, max_PT_jet, \
           d0_l0, d0_l1, d0_l2, n_jet, \
           n_btag, zsin_l0, zsin_l1, zsin_l2
    # no S to compute S/met
    # remaining is F_alpha and cannot compute max(1 - E/p)

@numba.jit(nopython=True)
def F_alpha(l0, l1, l2, met):
    trial = 100
    stepsize = 0.01
    successes = 0
    binweight = [0.000523209,0.0185422,0.0773298,0.1551565,0.19408925,0.1874,0.15113275,0.12033775,0.0721813,0.02330715]

    a1 = 91187.6**2
    b1 = 1776.82**2
    e1 = l1.E ** 2
    a2 = (80385**2)/2
    b2 = l2.E
    e2 =  -(l2.pz)
    for i in range(1, trial):
        alpha1 = 1 / (i*stepsize)
        c1 = alpha1 * (l0.E) * (l1.E)
        if ((alpha1 **2) * (l0.E) * (l0.E - b1))>=0:
            d1 = np.sqrt(alpha1**2 * l0.E * (l0.E - b1) * np.cos(l0.deltaangle(l1)))
        else:
            continue

        f1 = a1*a1*d1*d1*e1 - 4*a1*b1*d1*d1*e1 + 4*b1*b1*d1*d1*e1 - 4*b1*c1*c1*d1*d1 + 4*b1*d1*d1*d1*d1*e1
        g1 = c1**2 - (d1**2)*e1
        if np.logical_and((f1>=0),(g1!=0)):
            alpha2_1 = (np.sqrt(f1) + a1*c1 - 2*b1*c1) / (2*g1)
            alpha2_2 = (-np.sqrt(f1) + a1*c1 - 2*b1*c1) / (2*g1)
        else:
            continue

        if np.logical_and((alpha_2_1<1), (alpha2_2<1)):
            continue

        if alpha2_1 > 1:
            nuPx_1 = met.x - ((alpha1-1)*(l0.E * np.sin(l0.theta))) * np.cos(l0.phi) - ((alpha2_1-1)*(l1.E * np.sin(l1.theta) * np.cos(l1.phi)))
            nuPy_1 = met.y - ((alpha1 - 1) * (l0.E * np.sin(l0.theta))) * np.sin(l0.phi) - (
                        (alpha2_1 - 1) * (l1.E * np.sin(l1.theta) * np.sin(l1.phi)))

            c2_1 = nuPx_1**2 +nuPy_1**2
            d2_1 = - (l2.px*nuPx_1) - (l2.py*nuPy_1)

            f2_1 = a2**2*b2**2 - 2*a2*b2**2*d2_1 - b2**4*c2_1 + b2**2*c2_1*e2**2 + b2**2*d2_1**2
            g2_1 =  b2**2 - e2**2

            if np.logical_and(f2_1>=0, g2_1!=0):
                successes += binweight[(i-1)/10]

        if alpha2_2>1:
            nuPx_2 = met.x - ((alpha1-1)*(l0.E * np.sin(l0.theta))) * np.cos(l0.phi) - ((alpha2_2-1)*(l1.E * np.sin(l1.theta) * np.cos(l1.phi)))
            nuPy_2 = met.y - ((alpha1-1)*(l0.E * np.sin(l0.theta))) * np.sin(l0.phi) - ((alpha2_2-1)*(l1.E * np.sin(l1.theta) * np.sin(l1.phi)))

            c2_2 = nuPx_2**2 + nuPy_2**2
            d2_2 = - (l2.px*nuPx_2) - (l2.py*nuPy_2)

            f2_2 = a2**2*b2**2 - 2*a2*b2**2*d2_2 -  b2**4*c2_2 +  b2**2*c2_2*e2**2 + b2**2*d2_2**2
            g2_2 = b2**2 - e2**2

            if np.logical_and(f2_2>=0, g2_2!=0):
                successes += binweight[(i - 1) / 10]

    corrector = 0
    for j in range(0, 10):
        corrector = corrector + 10*binweight[j]

    retval = successes / (2*corrector)
    return retval


