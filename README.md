# PRISMA

Sensitivity analysis for various EFTs tested against SM only hypothesis

Uses Asimov data generated in MADGRAPH

Neural Networks are used to separate processes - our focus is on the VH to WWW channel

We also test validity of Higgs against bkg only hypothesis

Data here is fast simulated (does not include ttbar or system effects) - can modify code to include these if needed
  if using non-fast-sim data, include system effects in training data

selection_criteria.py : module full of functions for preselection of data
                        determines if an event falls into VH, WWW or background using Monte Carlo truths
                        if edited, ensure all cuts are applied to all 3 data types, as would be the case in real data analysis

pre_processing_.py : module to run datasets from root and lhe through the selection criteria
                     returns pandas dataframes of selected variables and the MC truths of events
                     saves csv files of datasets for SM and each EFT to your computer

plotting.py : robust function for plotting histograms for all the different input variables
              allows you to compare input variables from SM to EFT or from before and after being passd through multiclassifier

MLP_torch_optimiser.py : trains a variety of multiclassifier configurations
                         uses WandB logger - updates in real time so you can watch the progress of your models
                         can choose best model by looking at graphs in WandB, otherwise highest accuracy model is selected
                         

MLP_EFT_optimiser.py : trains a variety of model configurations to distinguish EFT and SM 
                       uses WandB logger - updates in real time so you can watch the progress of your models
                       can choose best model by looking at graphs in WandB, otherwise highest accuracy model is selected

output_nodes.py : finds and plots the output nodes of the various neural networks
                  makes the 'delta node' (WWW - VH node outputs) to reduce dimensionality of multiclassifier for analysis

analysis_.py : contains all the functions which will run analysis, using the functions from the other modules
               if you want to alter the analysis done using pyhf or change which plots are created look here

main.py : everything you need to perform the analysis (plots, print statements) is here
          has guard if not using top level environment
          generates data and trains models if you are running on your computer for the first time

requirements.txt : environment required to run project, run the following in your terminal:
                using pip
                pip install -r requirements.txt
            
                using Conda
                conda create --name <env_name> --file requirements.txt

HWW_WHFalpha1lowpt.cxx : C++ code adapted to obtain the F_alpha variable for the selection_criteria module


TLDR: Simply set up the environment and run main.py and you will get all the results :)
