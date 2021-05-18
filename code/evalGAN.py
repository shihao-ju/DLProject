# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:33:54 2021

@author: willi
"""

import os
import sys

path = os.path.abspath('../NYU_WIRELESS/mmwchanmod')
if not path in sys.path:
    sys.path.append(path)

import numpy as np
import pandas as pd
import pickle 

from resultsPlotGAN import resultsPlotGAN

#gan_out_fn = 'D:\\data\\output_paths_EpochFinal.csv'
gan_out_fn = 'D:\\data\\generated_paths.csv'
tr_data_fn = 'D:\\Wireless InSite\\Factory Warehouse\\Uplink\\factory_warehouse\\factory_warehouse.p'

nPath_gen = 20

# Get generated output csv from GAN
gen_samp_df = pd.read_csv(gan_out_fn)
gen_samp = np.array(gen_samp_df)[:,1:]

# Get training data file
dat_samp = pickle.load(open(tr_data_fn,"rb"))[1]

# Create plotter object
results = resultsPlotGAN(dat_samp, gen_samp)

# Evaluate
results.eval_path_loss(nPath_gen)
results.eval_angular_spread(nPath_gen)
results.eval_plos()
results.eval_rms_delay(nPath_gen)

