# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:56:41 2021

@author: willi
"""


import numpy as np
import os
import torch

from mmwchanmod.common.spherical import spherical_add_sub, cart_to_sph
from mmwchanmod.common.constants import PhyConst, AngleFormat

class chan3gppUAS(object):
    def __init__(self, scenario, hUT, fc, B, city):
    
        """
        3gpp UAV modeling class

        Parameters
        ----------
        scenario : str
            environemnt to be modeled. UMa-AV or UMi-AV
        tx_loc : [3 x 1] double
            position of transmitter
        rx_loc : [3 x 1] double
            position of receiver
        h_UT : double
            height of receiver
        fc : float
            carrier frequency
        bandwidth : float
            bandwidth
        d3D : float
            3D distance between TX and RX
        d2D: float
            Horizontal distance between TX and RX

        """   
        self.scenario = scenario
        #self.tx_loc = tx_loc
        #self.rx_loc = rx_loc
        self.h_UT = hUT #rx_loc[2]
        self.fc = fc
        self.bandwidth = B
        self.city = city
        #self.dvec = self.rx_loc - self.tx_loc
        
        
    def set_dist_vector(self,dvec):
        
        self.dvec = dvec
        
        self.d3D = np.sqrt( dvec[0]**2 + dvec[1]**2 + dvec[2]**2 )
        self.d2D = np.sqrt( dvec[0]**2 + dvec[1]**2 )
        
    def get_link_state_prob(self):
        
        if self.scenario == 'UMi-AV':           
            params = [18, 36, 294.05, -432.94, 233.98, -0.95]
        elif self.scenario == 'city':
            params_fn = 'trained_pLOS_3GPPUAS-'+self.city+'.pt'
            params = torch.load(params_fn)
            params = params.detach().numpy()
            
        if (self.h_UT <= 22.5):
            if self.d2D <= params[0]:
                self.p_LOS = 1
            else:
                self.p_LOS = params[0]/self.d2D + np.exp(-self.d2D/params[1])*(1-params[0]/self.d2D)
                
        elif self.h_UT > 22.5:
            p1 = params[4]*np.log10(self.h_UT)+params[5]
            d1 = np.max([params[2]*np.log10(self.h_UT)+params[3] , params[0]])
            
            if self.d2D <= d1:
                self.p_LOS = 1
            else:
                self.p_LOS = d1/self.d2D + np.exp(-self.d2D/p1)*(1-d1/self.d2D)
                    
                    
        elif self.scenario == 'UMa-AV':
            if (self.h_UT <= 22.5):
                if self.d2D <= 18:
                    self.p_LOS = 1   
                else:
                    if self.h_UT <= 13:
                        const = 0
                    else:
                        const = ((self.h_UT-13)/10)**1.5
                        
                    self.p_LOS = (18/self.d2D + np.exp(-self.d2D/63)\
                                  *(1-18/self.d2D))*(1+const*(5/4)\
                                  *(self.d2D/100)**3*np.exp(-self.d2D/150))
                                    
            elif (self.h_UT > 22.5) and (self.h_UT <= 100):
                p1 = 4300*np.log10(self.h_UT)-3800
                d1 = np.max([460*np.log10(self.h_UT)-700 , 18])
                
                if self.d2D <= d1:
                    self.p_LOS = 1
                else:
                    self.p_LOS = d1/self.d2D + np.exp(-self.d2D/p1)*(1-d1/self.d2D)
                    
            else:
                self.p_LOS = 1
                    
                    
                    
                    