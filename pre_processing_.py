import glob
from selection_criteria import *
from tqdm import tqdm
import pandas as pd

def pre_processing(filepath, root_signal, root_bkg, name):
    # function to produce a dataset with all necessary cuts and preselection
    # returned dataset has Type column for VH, WWW or bkg

    N_root_sig = getNeventsRoot(root_signal)
    N_root_bgk = getNeventsRoot(root_bkg) #saves computing it

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
            f_alpha = get_leptons(batch)
        # cannot compute max(1 - E/p)

        dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
                'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
                'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
                'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2, 'max_PT_jet': max_PT_jet,
                'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
                'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2':  invarM_l1l2,
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
             f_alpha = get_leptons(batch)
        # cannot compute max(1 - E/p)

        dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
                'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
                'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
                'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2, 'max_PT_jet': max_PT_jet,
                'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
                'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2':  invarM_l1l2,
                'm_lll': invarM, 'F_alpha': f_alpha}
        # make into dict    

        if df_WWW is not None:
            newdata_WWW = pd.DataFrame(dataset)
            df_WWW = pd.concat([df_WWW, newdata_WWW], axis=0, ignore_index=True)
            # concatenates new data with previous batch's data
        else:
            df_WWW = pd.DataFrame(dataset)
            # for the first round where there is no data

    print("Preselection efficiency for WWW events: %2.2f %%" % (100. * nselected_WWW / ntotal_WWW))

    df_WWW.insert(loc=0, column='Type', value=1) # WWW is event type 1

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
            f_alpha = get_leptons(batch)
        # cannot compute max(1 - E/p)

        dataset = {'PT_l0': PT_l0, 'PT_l1': PT_l1, 'PT_l2': PT_l2, 'met': met,
                'delR_l0l1': dR_l0l1, 'delR_l0l2': dR_l0l2, 'delR_l1l2': dR_l1l2,
                'delEta_l0l1': deltaeta_l0l1, 'delEta_l0l2': deltaeta_l0l2, 'delEta_l1l2': deltaeta_l1l2,
                'dPhi_MET_l0': dPhi_metl0, 'dPhi_MET_l1': dPhi_metl1, 'dPhi_MET_l2': dPhi_metl2, 'max_PT_jet': max_PT_jet,
                'mT_l0l1': transM_l0l1, 'mT_l0l2': transM_l0l2, 'mT_l1l2': transM_l1l2, 'sumPT': totalP,
                'm_l0l1': invarM_l0l1, 'm_l0l2': invarM_l0l2, 'm_l1l2':  invarM_l1l2,
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
    df_p.to_csv(f"{filepath}/df_preprocessed_{name}.csv",
                index=False)
    
def pre_processing_loadout(filepath):
    root_signal_SM = glob.glob(
        f"{filepath}/Samples/Signal/SM/WWW01j_000*.root")
    root_bkg = glob.glob(
        f"{filepath}/Samples/3l01j/3l01j_00*.root")
    name_SM = ""

    root_signal_cHq1_m1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cHq1_-1/WWW01j_00*.root")
    name_cHq1_m1 = "cHq1_-1"

    root_signal_cHq1_p1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cHq1_1/WWW01j_00*.root")
    name_cHq1_p1 = "cHq1_1"

    root_signal_cHq3_m1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cHq3_-1/WWW01j_00*.root")
    name_cHq3_m1 = "cHq3_-1"

    root_signal_cHq3_p1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cHq3_1/WWW01j_00*.root")
    name_cHq3_p1 = "cHq3_1"

    root_signal_cHW_m1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cHW_-1/WWW01j_00*.root")
    name_cHW_m1 = "cHW_-1"

    root_signal_cHW_p1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cHW_1/WWW01j_00*.root")
    name_cHW_p1 = "cHW_1"

    root_signal_cW_m1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cW_-1/WWW01j_00*.root")
    name_cW_m1 = "cW_-1"

    root_signal_cW_p1 = glob.glob(
        f"{filepath}/Samples/Signal/sig_cW_1/WWW01j_00*.root")
    name_cW_p1 = "cW_1"

    data_list = [[root_signal_SM, name_SM], [root_signal_cHq1_m1, name_cHq1_m1], [root_signal_cHq1_p1, name_cHq1_p1], [root_signal_cHq3_m1, name_cHq3_m1],
                [root_signal_cHq3_p1, name_cHq3_p1], [root_signal_cHW_m1, name_cHW_m1], [root_signal_cHW_p1, name_cHW_p1],
                [root_signal_cW_m1, name_cW_m1], [root_signal_cW_p1, name_cW_p1]]

    for i in range(0, len(data_list)):
        pre_processing(filepath, data_list[i][0], root_bkg, data_list[i][1])


