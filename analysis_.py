'''

Functions to run to get all the data for analysis
Uses functions from other modules so check everything is imported

'''

import numpy as np
import pandas as pd
import numba as nb
from selection_criteria import *
from plotting import plothist
from output_nodes import *

# Plotting
import matplotlib.pyplot as plt

# Data Formatting
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical

# Load Model
import joblib

# Classification Report
from sklearn.metrics import classification_report

# Scaling data
from sklearn.preprocessing import RobustScaler
import joblib
import torch

# Data Analysis
import pyhf
from scipy.stats import norm


def multiclassifier_results(filepath, best_MLP, full_graphs=False):
    # df of all data provided
    # can read in dfs from EFTs
    df = pd.read_csv(f"{filepath}/df_preprocessed_.csv")

    if full_graphs == True:
        #plot histograms for all data
        plothist(filepath, df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50)
        plothist(filepath, df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50)
        plothist(filepath, df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30)
        plothist(filepath, df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30)
        plothist(filepath, df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30)
        plothist(filepath, df, ["m_lll"], 1500., 50)
        plothist(filepath, df, ["sumPT"], 1500., 100)
        plothist(filepath, df, ["met"], 500., 100)
        plothist(filepath, df, ["max_PT_jet"], 500., 100)
        plothist(filepath, df, ["F_alpha"], 1., 20)

    # load trained ML model optimised for both VH and WWW separation
    filename = f"{filepath}/MLP_torch_{best_MLP}.sav"
    #filename = '/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/initial_model.sav'
    model = joblib.load(filename)

    # format data to use model on it
    x = df.drop('Type', axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))

    Y = df['Type'].values
    y_cat = to_categorical(Y, num_classes=3)

    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
    y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)

    # Get df post-training using the classifications predicted by the model 'y_test_class'
    y_pred = model(X)
    y_array = y.detach().numpy()
    y_pred = y_pred.detach().numpy()

    df_test = get_multiclassifier_nodes(filepath, y_pred, y_array)
    s_over_b(filepath, y_array, y_pred, '0', graph=True)
    s_over_b(filepath, y_array, y_pred, '1', graph=True)

    plot_nodes_multiclassifier(filepath, y_pred, y_array)

    y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data

    featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met',
                    'delR_l0l1', 'delR_l0l2', 'delR_l1l2',
                    'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2',
                    'dPhi_MET_l0', 'dPhi_MET_l1', 'dPhi_MET_l2','max_PT_jet',
                    'mT_l0l1', 'mT_l0l2', 'mT_l1l2', 'sumPT',
                    'm_l0l1', 'm_l0l2', 'm_l1l2',
                    'm_lll', 'F_alpha']

    x = np.array(sc.inverse_transform(x))
    # make df of results using predicted 'Type' classification
    df_test = pd.DataFrame(x, columns=featurenames)
    df_test.insert(loc=0, column='Type', value=y_pred_class)

    # TODO fix K fold CV for pytorch network
    # make df of post-training data
    #df_test = KFold_CV(df, model)

    weight_sig, weight_bkg, lumi = get_weights(filepath)

    if full_graphs == True:
        #histograms to compare input and output hists
        plothist(filepath, df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["m_lll"], 1500., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["sumPT"], 1500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["met"], 500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["max_PT_jet"], 500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
        plothist(filepath, df, ["F_alpha"], 1., 20, df2=df_test, weight2=weight_sig, name2='Post-ML')

    # cut on bkg output
    # make the WWW-VH data 
    df_test = df_test.drop('Type', axis=1)

    df_test.insert(loc=0, column='Pred Type', value=y_pred_class)
    df_test.insert(loc=0, column='Type', value=df['Type'].values)
    df_test_wo = df_test[df_test["Pred Type"] != 2] #remove everything predicted as bkg

    # format data to get node outputs
    x = df_test_wo.drop('Type', axis=1)
    x = x.drop('Pred Type', axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))
    Y = df_test_wo['Type'].values
    y_cat = to_categorical(Y, num_classes=3)
    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
    y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)
    y_pred = model(X)
    y_array = y.detach().numpy()
    y_pred = y_pred.detach().numpy()

    # get node outputs for everything predicted as signal (includes some bkg)
    array_list = plot_nodes_multiclassifier(filepath, y_pred, y_array, ret=True)
    array_VH = array_list[0]
    array_WWW = array_list[1]
    array_bkg = array_list[2]

    y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data using model
    y_truth_class = np.argmax(y_array, axis=1)  # classes of test data

    #get delta nodes ie WWW - VH node outputs eventwise
    delta_node_VH, delta_node_WWW, delta_node_bkg = get_delta_node(filepath, y_array, y_pred)

    # make bins of the delta nodes (here 10 bins)
    # ensure all bins have some signal and some bkg 
    hist_VH = binned_delta_node(delta_node_VH, weight_sig)
    hist_WWW = binned_delta_node(delta_node_WWW, weight_sig)
    hist_bkg = binned_delta_node(delta_node_bkg, weight_bkg)

    # bkg data includes all not of interest - here we look only at VH as interesting 
    bkg_data = [sum(x) for x in zip(hist_bkg, hist_WWW)]

    bkg_uncertainty_bkg = [np.sqrt((i / weight_bkg)) * weight_bkg for i in hist_bkg]
    bkg_uncertainty_WWW = [np.sqrt((i / weight_sig)) * weight_sig for i in hist_WWW]
    bkg_uncertainty = [np.hypot(x[0], x[1]) for x in zip(bkg_uncertainty_bkg, bkg_uncertainty_WWW)]

    # using pyhf for statistical analysis
    # test hypothesis of ONLY backkground - ie no VH signal (implies no Higgs)
    pyhf.set_backend("numpy")
    model = pyhf.simplemodels.uncorrelated_background(signal=(hist_VH), 
                                                    bkg=(bkg_data), 
                                                    bkg_uncertainty=(bkg_uncertainty))

    # Asimov data
    data = [sum(x) for x in zip(hist_bkg, hist_WWW, hist_VH)]

    data = data + model.config.auxdata

    best_fit_pars = pyhf.infer.mle.fit(data, model)
    print(f"initialization parameters for multiclassifier: {model.config.suggested_init()}")
    print(
        f"best fit parameters for multiclassifier:\
        \n * signal strength: {best_fit_pars[0]}\
        \n * nuisance parameters: {best_fit_pars[1:]}"
    )

    CLs_obs = [
        pyhf.infer.hypotest(0.0, data, model, test_stat='q0')
    ]

    #significance level
    sigma = norm.ppf(CLs_obs[0])

    return sigma


def get_p_val(mu, sigma, tmu_tilde):
    mean_over_sig = mu**2 / sigma**2
    if tmu_tilde <= mean_over_sig:
           F = 2 * norm.cdf(np.sqrt(tmu_tilde)) - 1

    elif tmu_tilde > mean_over_sig:
          fraction = (tmu_tilde + mean_over_sig) / ((2 * mu) / sigma)
          F = norm.cdf(np.sqrt(tmu_tilde)) + norm.cdf(fraction) - 1
    
    else:
          print('Invalid tmu_tilde')

    p = 1 - F 
    return p 

def test_significance_EFT(filepath, best_MLP, best_EFT_C, EFT_name):
    df_SM = pd.read_csv(f"{filepath}/df_preprocessed_.csv")
    df_SM.insert(loc=0, column="SM", value=1)

    weightEFT = get_weights_EFT(filepath, EFT_name)

    df_EFT = pd.read_csv(f"{filepath}/df_preprocessed_{EFT_name}.csv")
    df_EFT.insert(loc=0, column="SM", value=0)
    df_EFT = df_EFT[df_EFT['Type'] != 2]

    df = pd.concat([df_SM, df_EFT], axis=0, ignore_index=True)

    #run through multiclassifier, cut events predicted as bkg
    filename = f"{filepath}/MLP_torch_{best_MLP}.sav"
    model = joblib.load(filename)

    x = df.drop('Type', axis=1)
    x = x.drop('SM', axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))

    Y = df['Type'].values
    y_cat = to_categorical(Y, num_classes=3)

    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)

    # Get df post-training using the classifications predicted by the model 'y_test_class'
    y_pred = model(X)
    y_array = y.detach().numpy()
    y_pred = y_pred.detach().numpy()

    y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
    y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using model

    print('Classification of all data as VH / WWW / bkg')
    print(classification_report(y_truth_class, y_pred_class))

    featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met',
                    'delR_l0l1', 'delR_l0l2', 'delR_l1l2',
                    'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2',
                    'dPhi_MET_l0', 'dPhi_MET_l1', 'dPhi_MET_l2','max_PT_jet',
                    'mT_l0l1', 'mT_l0l2', 'mT_l1l2', 'sumPT',
                    'm_l0l1', 'm_l0l2', 'm_l1l2',
                    'm_lll', 'F_alpha']

    x = np.array(sc.inverse_transform(x))
    df_test = pd.DataFrame(x, columns=featurenames)
    df_test.insert(loc=0, column='Type', value=y_pred_class)
    df_test.insert(loc=0, column='Type truth', value=df['Type'])
    df_test.insert(loc=0, column='SM', value=df['SM'])

    # now run through classifier of EFT or SM 
    filenameEFT = f"{filepath}/MLP_EFT_{best_EFT_C}.sav"
    model_EFT = joblib.load(filenameEFT)

    df2 = df_test.drop("Type", axis=1)
    df2 = df2.drop('Type truth', axis=1)
    df2 = df2.dropna(axis=0)
    x = df2.drop("SM", axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))

    Y = df2['SM'].values

    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    y_pred = model_EFT(X)
    y_array = y.detach().numpy()
    y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using mode
    y_array = nb.typed.List([i[0] for i in y_array])
    y_pred = y_pred.detach().numpy()
    y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
    y_pred = nb.typed.List([i[0] for i in y_pred])

    x = np.array(sc.inverse_transform(x))
    df_test2 = pd.DataFrame(x, columns=featurenames)
    df_test2.insert(loc=0, column='Type', value=df_test['Type'])
    df_test2.insert(loc=0, column='Type truth', value=df_test['Type truth'])
    df_test2.insert(loc=0, column='SM truth', value=df_test['SM'])
    df_test2.insert(loc=0, column='SM', value=y_pred)

    #cut data predicted to be background by multiclassifier
    df_test2 = df_test2[df_test2['Type'] != 2]

    #make df of only truth VH and WWW events
    cut = (df_test2["Type truth"] == 0) | (df_test2["Type truth"] == 1) # types from MC truth
    df_VH_WWW = df_test2[cut]

    #make df of truth bkg events which were predicted to be signal
    cut_bkg = (df_test2["Type truth"] == 2)
    df_bkg = df_test2[cut_bkg]

    SM_cut =  (df_VH_WWW['SM truth'] == 1)
    EFT_cut =  (df_VH_WWW['SM truth'] == 0)
    SM_dist = df_VH_WWW[SM_cut]
    EFT_dist = df_VH_WWW[EFT_cut]

    weight_sig, weight_bkg, lumi = get_weights(filepath)  # weights & luminosity

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(0, 1, 40)
    ax.hist([SM_dist['SM'], EFT_dist['SM'], df_bkg['SM']], color=['purple', 'cyan', 'yellow'], bins=bins,
            weights=[np.full(len(SM_dist['SM']), weight_sig),
                        np.full(len(EFT_dist['SM']), weightEFT),
                        np.full(len(df_bkg['SM']), weight_bkg)],
            histtype='step', linewidth=5, label=["truth SM", "truth EFT", 'truth bkg'])
    ax.text(0.55, 0.95, "${\\cal L}=%3.0f$/fb" % lumi, transform=ax.transAxes)
    ax.set_xlabel(f"Network Output Distribution for Node {EFT_name}")
    #ax.set_yscale("log")
    ax.set_ylabel("Entries")
    ax.legend()
    plt.savefig(
        f"{filepath}/SM_node_w_bkg_{EFT_name}.png")
    plt.show()

    # check magnitudes of total events
    #sumSM = (len(SM_dist['SM'])) * weight_sig
    #sumbkg = (len(df_bkg['SM'])) * weight_bkg
    #sumEFT = (len(EFT_dist['SM'])) * weightEFT
    #print(sumSM, sumEFT, sumbkg)

    hist_SM = binned_EFT_node(SM_dist['SM'], weight_sig)
    hist_EFT = binned_EFT_node(EFT_dist['SM'], weightEFT)
    hist_bkg = binned_EFT_node(df_bkg['SM'], weight_bkg)

    bkg_uncertainty = [np.sqrt((i / weight_bkg)) * weight_bkg for i in hist_bkg]

    pyhf.set_backend("numpy", "minuit")
    model = pyhf.simplemodels.uncorrelated_background(signal=(hist_SM), 
                                                    bkg=(hist_bkg), 
                                                    bkg_uncertainty=(bkg_uncertainty))

    data = [sum(x) for x in zip(hist_EFT, hist_bkg)]

    data = data + model.config.auxdata

    fit_result, twice_nll_at_best_fit = pyhf.infer.mle.fit(data, model, return_uncertainties=True, return_fitted_val=True)
    pars, uncerts = fit_result.T

    print(f"initialization parameters for {EFT_name}: {model.config.suggested_init()}")
    print(
        f"best fit parameters for {EFT_name}:\
        \n * signal strength: {pars[0]}\
        \n * nuisance parameters: {pars[1:]}"
    )
    tmu_tilde = pyhf.infer.test_statistics.tmu_tilde(1.0, data, model, model.config.suggested_init(),
                                        model.config.suggested_bounds(),
                                            model.config.suggested_fixed())

    p_value = get_p_val(pars[0], uncerts[0], tmu_tilde)

    sigma = norm.ppf(1 - p_value)

    # unmute if you want individual log plots for each EFT
    #log_plot(data, model, twice_nll_at_best_fit, EFT_name)

    return p_value, data, model, twice_nll_at_best_fit, sigma

def log_plot(data, model, twice_nll_at_best_fit, data_list):
    
    test_mus = np.linspace(0, 10, 40)
    log_plot = []
    for test_poi in test_mus:
        bestfit_pars, twice_nll = pyhf.infer.mle.fixed_poi_fit( 
            test_poi, data, model, return_fitted_val=True)
        #print(-2 * model.logpdf(bestfit_pars, data) == twice_nll)
        log_plot.append(twice_nll - twice_nll_at_best_fit)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(test_mus, log_plot, label = str(data_list))
    ax.set_ylabel(r'$- 2ln(\frac{L(\mu, \hat{\hat{\theta}})}{L(\hat{\mu}, \hat{\theta})})$')
    ax.set_xlabel(r'$\mu$')
    plt.show()

def log_plot_multi(filepath, data, model, twice_nll_at_best_fit, data_list):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in range(0, len(data_list)):
        test_mus = np.linspace(0, 10, 40)
        log_plot = []
        for test_poi in test_mus:
            bestfit_pars, twice_nll = pyhf.infer.mle.fixed_poi_fit( 
                test_poi, data[i], model[i], return_fitted_val=True)
            #print(-2 * model.logpdf(bestfit_pars, data) == twice_nll)
            log_plot.append(twice_nll - twice_nll_at_best_fit[i])
        ax.plot(test_mus, log_plot, label = str(data_list[i]), marker='x')
    
    ax.legend()
    ax.set_ylabel(r'$- 2ln(\frac{L(\mu, \hat{\hat{\theta}})}{L(\hat{\mu}, \hat{\theta})})$')
    ax.set_xlabel(r'$\mu$')
    plt.savefig(f'{filepath}/logplot_all_EFTs.png')
    plt.show()