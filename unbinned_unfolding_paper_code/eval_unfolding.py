import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import yaml
import json
import h5py as h5
import gzip
import pickle
import argparse 
from tabulate import tabulate

# Locals 
from omnifold import DataLoader, MultiFold, MLP, PET, SetStyle, HistRoutine
# For IBU and plotting (source: https://github.com/ericmetodiev/OmniFold/blob/master/modplot.py)
import modplot
from hist_utils import obs, calc_obs, hist_style, gen_style, truth_style, omnifold_style, multifold_style, ibu_style
import ibu

def expit(x):
    return 1. / (1. + np.exp(-x))

def reweight(events,model,batch_size=None):
    f = expit(model.predict(events,batch_size=batch_size))
    weights = f / (1. - f)  # this is the crux of the reweight, approximates likelihood ratio
    weights = np.nan_to_num(weights[:,0],posinf=1)
    return weights

def compute_delta(p, q):
    numerator = (p-q)**2
    denominator = p+q+1e-50
    delta = 1e3*(1/2)*np.sum(numerator/denominator)
    return delta

def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions using trained PET model")
    parser.add_argument("--data_dir", type=str, default="/pscratch/sd/a/aelabd/omnifold_examples/ryans_way/OmniLearn/data/", help="Folder containing input files")
    parser.add_argument("--save_dir", type=str, default="/pscratch/sd/a/aelabd/omnifold_examples/ryans_way/unbinned_unfolding/unbinned_unfolding_paper_code/output", help="Folder containing input files")
    parser.add_argument("--of_run_id", type=int, default=None, help="Unique identifier for OmniFold training run; necessary in order to retrieve weights")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'serif'

    # IBU hyperparameter
    N_ITER_IBU = 3

    ############################### Load data ###############################
    RUN_DIR = os.path.join(args.save_dir, f"run_{args.of_run_id}")
    with open(os.path.join(RUN_DIR, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    OBSERVABLES = ['Mass', 'Width', 'Mult', 'SDMass', 'zg', 'Tau21']

    # Need this for IBU
    SYNTH_DATA_PATH = os.path.join(args.data_dir, "test_pythia.h5")
    synth_all = h5.File(SYNTH_DATA_PATH, 'r')
    synth_gen_particles = synth_all['gen'][:config['num_data']]
    synth_gen_observables = synth_all['gen_subs'][:config['num_data']]
    synth_data_particles = synth_all['reco'][:config['num_data']]
    synth_data_observables = synth_all['reco_subs'][:config['num_data']]

    synth_gen_observables = {k: synth_gen_observables[:,i] for i, k in enumerate(OBSERVABLES)}
    synth_data_observables = {k: synth_data_observables[:,i] for i, k in enumerate(OBSERVABLES)}


    NATURE_DATA_PATH =  os.path.join(args.data_dir, "test_herwig.h5") 
    nature_all = h5.File(NATURE_DATA_PATH, 'r')
    nature_gen_particles = nature_all['gen'][:config['num_data']]
    nature_gen_observables = nature_all['gen_subs'][:config['num_data']]
    nature_data_particles = nature_all['reco'][:config['num_data']]
    nature_data_observables = nature_all['reco_subs'][:config['num_data']]

    nature_gen_observables = {k: nature_gen_observables[:,i] for i, k in enumerate(OBSERVABLES)}
    nature_data_observables = {k: nature_data_observables[:,i] for i, k in enumerate(OBSERVABLES)}



    ############################### Compute GEN, DATA, and TRUTH histograms ###############################
    for obkey, ob in obs.items():
        ob['genobs'] = synth_gen_observables[obkey]
        ob['simobs'] = synth_data_observables[obkey]
        ob['truthobs'] = nature_gen_observables[obkey]
        ob['dataobs'] = nature_data_observables[obkey]

        # setup bins
        ob['bins_det'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_det']+1)
        ob['bins_mc'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_mc']+1)
        ob['midbins_det'] = (ob['bins_det'][:-1] + ob['bins_det'][1:])/2
        ob['midbins_mc'] = (ob['bins_mc'][:-1] + ob['bins_mc'][1:])/2
        ob['binwidth_det'] = ob['bins_det'][1] - ob['bins_det'][0]
        ob['binwidth_mc'] = ob['bins_mc'][1] - ob['bins_mc'][0]

        # get the histograms of GEN, DATA, and TRUTH level observables
        ob['genobs_hist'] = np.histogram(ob['genobs'], bins=ob['bins_mc'], density=True)[0]
        ob['data_hist'] = np.histogram(ob['dataobs'], bins=ob['bins_det'], density=True)[0]
        ob['truth_hist'], ob['truth_hist_unc'] = modplot.calc_hist(ob['truthobs'], bins=ob['bins_mc'], 
                                                                density=True)[:2]

        # compute (and normalize) the response matrix between GEN and SIM
        ob['response'] = np.histogram2d(ob['simobs'], ob['genobs'], bins=(ob['bins_det'], ob['bins_mc']))[0]
        ob['response'] /= (ob['response'].sum(axis=0) + 10**-50)

    ############################### Compute IBU ###############################
    print(f"Computing IBU...")
    for obkey, ob in obs.items():
        # perform iterative Bayesian unfolding
        ob['ibu_phis'] = ibu.ibu(ob['data_hist'], ob['response'], ob['genobs_hist'], 
                            ob['binwidth_det'], ob['binwidth_mc'], it=N_ITER_IBU)
        ob['ibu_phi_unc'] = ibu.ibu_unc(ob, it=N_ITER_IBU, nresamples=25)

        print('Done with', obkey)

    
    ############################### Load OmniFold weights ###############################
    of_weights_dir = os.path.join(RUN_DIR, "predictions")
    with gzip.open(os.path.join(of_weights_dir, "test_pythia_final_reweight.pickle.gz"), "r") as f:
        of_weights = pickle.load(f)
    of_weights = of_weights["PET_weights"]

    ################################ Plot, save figures ###############################
    for i,(obkey,ob) in enumerate(obs.items()):
        
        # get the styled axes on which to plot
        fig, [ax0, ax1] = modplot.axes(**ob, figsize=(6,6))
        if ob.get('yscale') is not None:
            ax0.set_yscale(ob['yscale'])
            
        # Plot the Different Distributions of the Observable
        # plot the "data" histogram of the observable
        ax0.hist(ob['dataobs'], bins=ob['bins_det'], color='black', label='``Data\"', **hist_style)

        # plot the "truth" histogram of the observable
        ax0.fill_between(ob['midbins_mc'], ob['truth_hist'], **truth_style)

        # plot the IBU distribution
        ax0.plot(ob['midbins_mc'], ob['ibu_phis'][N_ITER_IBU], **ibu_style, label='IBU ' + ob['symbol'])


        # plot the OmniFold distribution
        of_histgen, of_histgen_unc = modplot.calc_hist(ob['genobs'], weights=of_weights, 
                                                    bins=ob['bins_mc'], density=True)[:2]
        ax0.plot(ob['midbins_mc'], of_histgen, **omnifold_style, label='OmniFold')


        # Plot the Ratios of the OmniFold and IBU distributions to truth (with statistical uncertainties)
        ibu_ratio = ob['ibu_phis'][N_ITER_IBU]/(ob['truth_hist'] + 10**-50)
        of_ratio = of_histgen/(ob['truth_hist'] + 10**-50)
        ax1.plot([np.min(ob['midbins_mc']), np.max(ob['midbins_mc'])], [1, 1], '-', color='green', lw=0.75)
        
        # ratio uncertainties
        truth_unc_ratio = ob['truth_hist_unc']/(ob['truth_hist'] + 10**-50)
        ibu_unc_ratio = ob['ibu_phi_unc']/(ob['truth_hist'] + 10**-50)
        of_unc_ratio = of_histgen_unc/(ob['truth_hist'] + 10**-50)
        
        ax1.fill_between(ob['midbins_mc'], 1 - truth_unc_ratio, 1 + truth_unc_ratio, 
                        facecolor=truth_style['facecolor'], zorder=-2)
        ax1.errorbar(ob['midbins_mc'], ibu_ratio, xerr=ob['binwidth_mc']/2, yerr=ibu_unc_ratio, 
                                                color=ibu_style['color'], **modplot.style('errorbar'))
        ax1.errorbar(ob['midbins_mc'], of_ratio, xerr=ob['binwidth_mc']/2, yerr=of_unc_ratio, 
                                                color=omnifold_style['color'], **modplot.style('errorbar'))

        # legend style and ordering
        loc, ncol = ob.get('legend_loc', 'upper right'), ob.get('legend_ncol', 2)
        order = [0, 2, 1, 3] if ncol==2 else [0, 1, 2, 3]
        modplot.legend(ax=ax0, frameon=False, order=order, loc=loc, ncol=ncol)

        # stamp to put on the plots
        modplot.stamp(*ob['stamp_xy'], delta_y=0.06, ax=ax0,
                    line_0=r'$\mathbf{D/T}$: Herwig 7.1.5 default',
                    line_1=r'$\mathbf{S/G}$: Pythia 8.243 tune 26',
                    line_2=r'Delphes 3.4.2 CMS Detector',
                    line_3=r'$Z$+jet: $p_T^Z>200$ GeV, $R=0.4$')
        
        # save plot (by default in the same directory as this notebook).
        # If running on binder, the plot can be accessed by first going to the jupyter file browser
        # (which itself can be accessed by copying the URL of this notebook and removing the name of the notebook
        # after the final "/"), selecting the square next to the name of the plot, and clicking "Download".
        fig_dir = os.path.join(RUN_DIR, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir,f"{obkey}.pdf"), bbox_inches='tight')
        plt.close()

    ################################ Compute triangular discriminant ###############################
    
    OBSERVABLES = ['Mass', 'Mult', 'Width',  'SDMass', 'Tau21', 'zg'] # reordered to match Table 1 in the paper

    # Compute the 'true' probability distributions, q
    q_dict = {}
    for k in OBSERVABLES:
        q_dict[k] = obs[k]["truth_hist"].copy()/sum(obs[k]["truth_hist"])

    # Compute the IBU probability distribution, p_ibu
    p_ibu_dict = {}
    for k in OBSERVABLES:
        p_ibu_dict[k] = obs[k]["ibu_phis"][-1].copy()/sum(obs[k]["ibu_phis"][-1])

    # Compute the OmniFold probability distribution, p_of
    p_of_dict = {}
    for k in OBSERVABLES:
        of_histgen, of_histgen_unc = modplot.calc_hist(obs[k]['genobs'], weights=of_weights, 
                                                        bins=obs[k]['bins_mc'], density=True)[:2]
        p_of_dict[k] = of_histgen.copy()/sum(of_histgen)
    
    # Compute triangular disciminant
    delta_dict = {"IBU": {},  "OmniFold": {}}
    for k in OBSERVABLES:
        delta_dict['IBU'][k] = compute_delta(p_ibu_dict[k], q_dict[k])
        delta_dict['OmniFold'][k] = compute_delta(p_of_dict[k], q_dict[k])

    # Display/save
    headers = ["Method", "m", "M", "w", "SDMass", "t21", "zg"]
    tri_disc_data = [
        ["OmniFold"],
        ["IBU"],
    ]
    tri_disc_data[0].extend(delta_dict["OmniFold"].values())
    tri_disc_data[1].extend(delta_dict["IBU"].values())
    print(tabulate(tri_disc_data, headers=headers, tablefmt="grid"))



    
    with open(os.path.join(RUN_DIR, "triangular_discriminants.json"), 'w', encoding='utf-8') as f:
        json.dump(delta_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()