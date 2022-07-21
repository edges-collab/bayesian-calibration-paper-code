# Bayesian Calibration

This repo contains code and notebooks that perform Bayesian fits,
and produce plots, for Murray+22 ("A Bayesian Calibration Framework for EDGES").
There is a [different repository](https://github.com/edges-collab/bayesian-calibration)
that has a broader scope than this one (it has more development/research/open ended stuff).


## Repo Layout

The repo contains the following contents:

* `alan-data/`: this contains all the calibration data from Bowman+2018 that
   is required for this work. This includes the calibratd S11's, beam correction,
   losses, and final calibrated/averaged sky spectrum. Some of these are used
   directly in our re-calibration here, and others are used as a check. All of these
   products can be obtained by running code from the [alans-pipeline](https://github.com/edges-collab/alans-pipeline)
   repo. 
* `calfiles/`: the lab-calibration file produced by `edges-cal` based on settings
   that attempt to match Bowman+2018 as closely as possible. The file here 
   is created in the `getting_2015_calibration.ipynb` notebook. 
* `outputs/` and `plots/`: these will have to be created if you are trying to
   run this code. They will end up containing polychord output chains, and the
   plots for the paper, respectively.
* **Notebooks**: The primary resource in this repo is the notebooks. They roughly 
  correspond to different stages of the analysis (or sections of the paper), in 
  the following order:
  * `raw_data_assumptions`: work exploring assumptions on our data model, like
     Gaussianity and independence of frequency channels. Figures from here
     appear in the section  4.1 "The Noise-Wave Formalism"
  * `getting_2015_calibration`: this compares the calibration done with `edges-cal`
     to that done in B18, and produces our "calfile" that is used for all calibration
     throughout the repo.
  * `precal`: this notebook examines the calibration-only likelihood, typically not
     using full polychord chains, but just max-likelihood techniques. It examines
     the bias of the likelihood model compared to the iterative method.
  * `calibration`: examines the polychord chains arising from the calibration-only
     likelihood.
  * `predata`: plots of sky-data related quantities that don't require the full
    polychord runs (eg. the loss function)
  * `sky-data`: plots of the posteriors from models including sky data (either jointly
    with calibration or in isolation).
* **Python modules**: we include some python utility modules that are shared between
  notebooks:
  * `alan_data.py`: reads data from `alan-data/`
  * `alan_data_utils.py`: using the data from `alan-data/`, provides utilities to 
    set up calibration, and create likelihoods.
  * `analyse_mcmc.py`: some utilities for reading/analysing the output MCMC chains 
    from polychord
  * `mcmc_utils.py`: utilities for batch management of MCMC runs, according to the 
    database layout we use.
  * `notebook_utils.py`: simple setting up for the various notebooks (eg. setting
    up matplotlib to have correct settings for "paper" plots).
* **Python Scripts**: we include several python scripts that are used to actually
  run the MCMC with different settings. They each share some configuration, which 
  is set up in `run_mcmc_utils.py`. All scripts are called `run_alan_XXX_mcmc.py`.
  Each corresponds to a different "stage" of analysis (either precal, pure calibration,
  sky-data, or sky-data in isolation).
  