"""
param_est_3gppUAS.py:  classes for parameterizing 3gpp model
"""

import numpy as np
#import math
import torch

from mmwchanmod.datasets.download import get_dataset
from mmwchanmod.common.constants import LinkState


class param_estimator_3gppUAS(object):
    def __init__(self,param,city,cell,hbs):
        """
        3gpp UAS parameter modeling class

        Parameters
        ----------
        param : str
            parameter to estimate. Options:
                - pLOS
        city : str
            city data to use for re-parameterization
        rx_type : int
            gNB type to use:
                - 0 for aerial
                - 1 for terrestrial
        """
        self.param = param
        self.city = city
        self.rx_type = cell
        self.hbs = hbs
        
    def load_data(self):
        if self.param == 'pLOS':
            
            print('Processing %s' % self.city)
            ds_name = ('uav_%s' % self.city)
            ds = get_dataset(ds_name)
            mod_name = ('uav_%s' % self.city)
            cfg = ds['cfg']
            
            self.dat_tr = ds['train_data']
            self.dvec = self.dat_tr['dvec']
            
    def format_data(self):
            
        # Get the link state
        link_state = self.dat_tr['link_state']
        link_exists_ind = (link_state == LinkState.los_link) | (link_state == LinkState.nlos_link)
        
        
        # Get horizontal and vertical distances
        dx = np.sqrt(self.dvec[:,0]**2 + self.dvec[:,1]**2)
        dz = self.dvec[:,2]
        
        # Get labels for where a link exists
        I = np.where((self.dat_tr['rx_type'] == self.rx_type) & link_exists_ind)[0]
        
        # Index the distance arrays
        self.dx_tr = dx[I]
        self.dz_tr = dz[I]

        # Compute UT height (3GPP standard)
        #self.hut = self.dz_tr + np.abs(min(self.dz_tr)) + self.hbs
        
        self.hut = self.dz_tr + self.hbs
        
        # Form datasets
        self.xtr = torch.tensor([np.log10(self.hut), self.dx_tr, self.hut])
        self.ytr = torch.tensor(link_state[I]-1)
            
        
class layer_3gppUAS_LOS(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate parameters to estimate and assign 
        them as member parameters.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([18, \
                                                        36, \
                                                        294.05, \
                                                        -432.94, \
                                                        233.98, \
                                                        -0.95]))

    def forward(self, x):
        """
        In the forward function we accept input data and we return an output
        probability
        
        """
        # Initialize tensors
        nsamp = (x.size())[1]
        d1 = torch.zeros(nsamp)
        p1 = torch.zeros(nsamp)
        
        # Find indices where hut is greater than or less than 22.5m
        I0 = torch.where(x[2] <= 22.5)
        I1 = torch.where(x[2] > 22.5)
        
        # Define d1 and p1 accordingly
        d1[I0] = self.weights[0]
        d1[I1] = torch.max(self.weights[2]*x[0][I1] + self.weights[3],self.weights[0])
        
        p1[I0] = self.weights[1]
        p1[I1] = self.weights[4]*x[0][I1] + self.weights[5]
        
        # Compute plos for every sample
        return torch.min(torch.ones(nsamp), d1/x[1] + torch.exp(-x[1]/p1)*(1-d1/x[1]))