# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:23:04 2021

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math

import os
import sys

path = os.path.abspath('..')
if not path in sys.path:
    sys.path.append(path)

from mmwchanmod.common.spherical import spherical_add_sub, cart_to_sph
from mmwchanmod.common.constants import PhyConst, AngleFormat
from mmwchanmod.common.constants import LinkState

class resultsPlotGAN(object):
    def __init__(self,training_data,generated_data):
        """
        GAN results evaluation class
    
        Parameters
        ----------
        training_data : pandas dataframe
            data used to train the GAN
        generated_data : array, float
            output of the GAN ("fake" data)
            
        """   
        self.training_data = training_data
        self.generated_data = generated_data
        
    def eval_plos(self):
        """
        Plots the probability of LOS as a function of distance

        """
        # Get the test data vector
        dvec = self.training_data['dvec']
        dx = np.sqrt(dvec[:,0]**2 + dvec[:,1]**2)
        dz = dvec[:,2]
        
        # Get the link state data
        link_state = self.training_data['link_state']
        los = (link_state == LinkState.los_link)
        
        # Extract the correct points    
        I0 = np.where(los)[0]
        
        # Set plotting limits
        xlim = np.array([np.min(dx), np.max(dx)])
        zlim = np.array([np.min(dz), np.max(dz)])
        
        # Compute the empirical probability
        H0, xedges, zedges = np.histogram2d(dx[I0],dz[I0],bins=[20,2],range=[xlim,zlim])
        Htot, xedges, zedges = np.histogram2d(dx,dz,bins=[20,2],range=[xlim,zlim])
        prob_ts = H0 / np.maximum(Htot,1)
        prob_ts = np.flipud(prob_ts.T)
        
        # Plot the results
        plt.subplot(2,2,1)
        plt.imshow(prob_ts,aspect='auto',\
               extent=[np.min(xedges),np.max(xedges),np.min(zedges),np.max(zedges)],\
               vmin=0, vmax=1)
        plt.title('Data')
        plt.ylabel('Elevation (m)')
        
        #plt.xticks([])

        plt.xlabel('Horiz (m)')    
        
        
    def eval_path_loss(self, npath_gen):
        """
        Plots the CDF of the path loss of both data and generated samples

        """
        
        # Flatten the array of ray traced path losses
        path_loss_dat = np.ndarray.flatten(self.training_data['nlos_pl'][:,:24])
        
        # Get array of the generated path losses
        path_loss_gen = []
        for n in range(npath_gen):
            path_loss_gen = np.append(path_loss_gen,self.generated_data[:,4+n*6])
        
        # Plot the CDF
        p1 = len(path_loss_dat)
        p2 = len(path_loss_gen)
        
        plt.plot(np.sort(path_loss_dat),np.arange(p1)/p1,\
                 label = "Data Samples")
        plt.plot(np.sort(path_loss_gen),np.arange(p2)/p2,\
                 label = "Generated Samples")
        
        plt.xlabel('Path Loss [dB]')
        plt.ylabel('CDF')
        
        plt.legend()
        plt.grid()
        plt.show()
    
        
    def eval_angular_spread(self,npath_gen):
        """
        Computes and plots the angular spread of each AoD, ZoD, AoA, and ZoA 
        for both data and generated samples.
        Computation of angular spread taken from 3GPP 38.901 Annex A.

        """
        # Get the sample (true) angular data
        nDatSamp = len(self.training_data['nlos_ang'])
        nPathsSamp = 24

        ang_dat = self.training_data['nlos_ang'][:,:nPathsSamp,:]
        
        # Get the generated (fake) angular data
        nGenSamp = len(self.generated_data)
        ang_gen = np.zeros([nGenSamp,npath_gen,4])
        for n in range(npath_gen):
            ang_gen[:,n,:] = self.generated_data[:,6+n*6:10+n*6]
            
        # Get the sample path loss data
        path_loss_dat = self.training_data['nlos_pl'][:,:nPathsSamp]
        
        # Get the generated path loss data
        path_loss_gen = np.zeros([nGenSamp,npath_gen])
        for n in range(npath_gen):
            path_loss_gen[:,n] = self.generated_data[:,4+n*6]
        
        # Convert gains to linear scale
        path_loss_dat = 10**(-0.05*path_loss_dat)
        path_loss_gen = 10**(-0.05*path_loss_gen)
        
        # Separate data angles, convert to radians
        aod_dat = ang_dat[:,:,0]*math.pi/180
        zod_dat = ang_dat[:,:,1]*math.pi/180
        aoa_dat = ang_dat[:,:,2]*math.pi/180
        zoa_dat = ang_dat[:,:,3]*math.pi/180
        
        # Compute angular spreads of the data angles
        spread_aod_dat = np.zeros(nDatSamp)
        spread_zod_dat = np.zeros(nDatSamp)
        spread_aoa_dat = np.zeros(nDatSamp)
        spread_zoa_dat = np.zeros(nDatSamp)
        for n in range(nDatSamp):
            spread_aod_dat[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*aod_dat[n,:]),path_loss_dat[n,:]))\
                                        /np.sum(path_loss_dat[n,:]))))
                
            spread_zod_dat[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*zod_dat[n,:]),path_loss_dat[n,:]))\
                                        /np.sum(path_loss_dat[n,:]))))
            spread_aoa_dat[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*aoa_dat[n,:]),path_loss_dat[n,:]))\
                                        /np.sum(path_loss_dat[n,:]))))
                
            spread_zoa_dat[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*zoa_dat[n,:]),path_loss_dat[n,:]))\
                                        /np.sum(path_loss_dat[n,:]))))
        
        # Separate generated angles, convert to radians
        aod_gen = ang_gen[:,:,0]*math.pi/180
        zod_gen = ang_gen[:,:,1]*math.pi/180
        aoa_gen = ang_gen[:,:,2]*math.pi/180
        zoa_gen = ang_gen[:,:,3]*math.pi/180
        
        # Compute angular spreads of the generated angles
        spread_aod_gen = np.zeros(nGenSamp)
        spread_zod_gen = np.zeros(nGenSamp)
        spread_aoa_gen = np.zeros(nGenSamp)
        spread_zoa_gen = np.zeros(nGenSamp)
        for n in range(nGenSamp):
            spread_aod_gen[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*aod_gen[n,:]),path_loss_gen[n,:]))\
                                        /np.sum(path_loss_gen[n,:]))))
                
            spread_zod_gen[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*zod_gen[n,:]),path_loss_gen[n,:]))\
                                        /np.sum(path_loss_gen[n,:]))))                
                                                           
            spread_aoa_gen[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*aoa_gen[n,:]),path_loss_gen[n,:]))\
                                        /np.sum(path_loss_gen[n,:]))))
                
            spread_zoa_gen[n] = np.sqrt(-2*np.log(np.abs(np.sum(np.multiply(\
                                np.exp(1j*zoa_gen[n,:]),path_loss_gen[n,:]))\
                                        /np.sum(path_loss_gen[n,:]))))
        
        
        # Plot CDFs of the angular spreads
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        
        # Azimuth of Departure
        p1 = len(spread_aod_dat)
        p2 = len(spread_aod_gen)
        
        axs[0,0].plot(np.sort(spread_aod_dat),np.arange(p1)/p1,\
                 label = "Dat")
        axs[0,0].plot(np.sort(spread_aod_gen),np.arange(p2)/p2,\
                 label = "Gen")
        
        axs[0,0].set_xlabel('Angular Spread $\phi_{tx}$')
        axs[0,0].set_ylabel('CDF')
        
        axs[0,0].legend()
        axs[0,0].grid()
        
        # Zenith of Departure
        p1 = len(spread_zod_dat)
        p2 = len(spread_zod_gen)
        
        axs[0,1].plot(np.sort(spread_zod_dat),np.arange(p1)/p1,\
                 label = "Dat")
        axs[0,1].plot(np.sort(spread_zod_gen),np.arange(p2)/p2,\
                 label = "Gen")
        
        axs[0,1].set_xlabel('Angular Spread $\\theta_{tx}$')
        axs[0,0].set_ylabel('CDF')
        
        axs[0,1].legend()
        axs[0,1].grid()
        
        # Azimuth of Arrival
        p1 = len(spread_aoa_dat)
        p2 = len(spread_aoa_gen)
        
        axs[1,0].plot(np.sort(spread_aoa_dat),np.arange(p1)/p1,\
                 label = "Dat")
        axs[1,0].plot(np.sort(spread_aoa_gen),np.arange(p2)/p2,\
                 label = "Gen")
        
        axs[1,0].set_xlabel('Angular Spread $\phi_{rx}$')
        axs[1,0].set_ylabel('CDF')
        
        axs[1,0].legend()
        axs[1,0].grid()
        
        # Zenith of Arrival
        p1 = len(spread_zoa_dat)
        p2 = len(spread_zoa_gen)
        
        axs[1,1].plot(np.sort(spread_zoa_dat),np.arange(p1)/p1,\
                 label = "Dat")
        axs[1,1].plot(np.sort(spread_zoa_gen),np.arange(p2)/p2,\
                 label = "Gen")
        
        axs[1,1].set_xlabel('Angular Spread $\\theta_{rx}$')
        axs[1,1].set_ylabel('CDF')
        
        axs[1,1].legend()
        axs[1,1].grid()
        
        plt.show()
        

    def eval_rms_delay(self,npath_gen):
        """
        Compares the RMS delay CDF of the true data to generated samples

        Parameters
        ----------
        npath_gen : int
            Number of generated paths to use

        Returns
        -------
        None.

        """
        nDatSamp = len(self.training_data['nlos_dly'])
        nPathsSamp = 24

        # Get the data sample (true) abosolute propagation delay data
        dly_dat = self.training_data['nlos_dly'][:,:nPathsSamp]
        
        # Get the generated (fake) abosolute propagation delay data
        nGenSamp = len(self.generated_data)
        dly_gen = np.zeros([nGenSamp,npath_gen])
        for n in range(npath_gen):
            dly_gen[:,n] = self.generated_data[:,5+n*6]
        
        # Get the sample path loss data
        path_loss_dat = self.training_data['nlos_pl'][:,:nPathsSamp]
        
        # Get the generated path loss data
        path_loss_gen = np.zeros([nGenSamp,npath_gen])
        for n in range(npath_gen):
            path_loss_gen[:,n] = self.generated_data[:,4+n*6]
        
        # Convert gains to linear scale
        path_loss_dat = 10**(-0.05*path_loss_dat)
        path_loss_gen = 10**(-0.05*path_loss_gen)
        
        # Compute mean delay for each link in both true and fake samples
        mean_dly_dat = np.sum(np.multiply(dly_dat,path_loss_dat),axis=1)\
                        /np.sum(path_loss_dat,axis=1)
        mean_dly_gen = np.sum(np.multiply(dly_gen,path_loss_gen),axis=1)\
                        /np.sum(path_loss_gen,axis=1)
        
        # Compute root mean square delay for data samples
        rms_dly_dat = np.zeros([nDatSamp])
        for n in range(nDatSamp):
            rms_dly_dat[n] = np.sqrt(np.sum(np.multiply(path_loss_dat[n,:],\
                             np.square(dly_dat[n,:]-mean_dly_dat[n])))\
                                        /np.sum(path_loss_dat[n,:]))
                                        
        # Compute root mean square delay for generated samples
        rms_dly_gen = np.zeros([nGenSamp])
        for n in range(nGenSamp):
            rms_dly_gen[n] = np.sqrt(np.sum(np.multiply(path_loss_gen[n,:],\
                             np.square(dly_gen[n,:]-mean_dly_gen[n])))\
                                        /np.sum(path_loss_gen[n,:]))     
        
        # Plot the CDF
        plt.figure()

        p1 = len(rms_dly_dat)
        p2 = len(rms_dly_gen)
        
        plt.plot(np.sort(rms_dly_dat*1e6),np.arange(p1)/p1,\
                 label = "Data Samples")
        plt.plot(np.sort(rms_dly_gen*1e6),np.arange(p2)/p2,\
                 label = "Generated Samples")
        
        plt.xlabel('RMS Delay [$\mu$s]')
        plt.ylabel('CDF')
        
        plt.legend()
        plt.grid()
        plt.show()        
        

    def transform_ang(self, dvec, nlos_ang, nlos_pl):
        """
        Performs the transformation on the angle data

        Parameters
        ----------
        dvec : (nlink,ndim) array
            Vectors from cell to UAV for each link
        nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees
        nlos_pl : (nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of pl_max indicates no path

        Returns
        -------
        Xang : (nlink,nangle*npaths_max)
            Tranformed angle coordinates

        """
                
        # Compute the LOS angles
        r, los_aod_phi, los_aod_theta = cart_to_sph(dvec)
        r, los_aoa_phi, los_aoa_theta = cart_to_sph(-dvec)
        
        # Get the NLOS angles
        nlos_aod_phi   = nlos_ang[:,:,AngleFormat.aod_phi_ind]
        nlos_aod_theta = nlos_ang[:,:,AngleFormat.aod_theta_ind]
        nlos_aoa_phi   = nlos_ang[:,:,AngleFormat.aoa_phi_ind]
        nlos_aoa_theta = nlos_ang[:,:,AngleFormat.aoa_theta_ind]
        
        # Rotate the NLOS angles by the LOS angles to compute
        # the relative angle        
        aod_phi_rel, aod_theta_rel = spherical_add_sub(\
            nlos_aod_phi, nlos_aod_theta,\
            los_aod_phi[:,None], los_aod_theta[:,None])
        aoa_phi_rel, aoa_theta_rel = spherical_add_sub(\
            nlos_aoa_phi, nlos_aoa_theta,\
            los_aoa_phi[:,None], los_aoa_theta[:,None])            
            
        # Set the relative angle on non-existent paths to zero
        I = (nlos_pl > 150+0.01)
        aod_phi_rel = aod_phi_rel*I
        aod_theta_rel = aod_theta_rel*I
        aoa_phi_rel = aoa_phi_rel*I
        aoa_theta_rel = aoa_theta_rel*I
                                        
        # Stack the relative angles and scale by 180
        Xang = np.hstack(\
            (aoa_phi_rel/180, aoa_theta_rel/180,\
             aod_phi_rel/180, aod_theta_rel/180))
        
        return Xang
    
    def plot_ang_dist(self,ax,dvec,nlos_ang,nlos_pl,iang,pl_tol=30,dmax=1000,generated=False):
        
        #angle_dist_dat = np.ndarray.flatten(self.training_data['nlos_ang'][:,:10])
        

        """
        Plots the conditional distribution of the relative angle.
        
        Parameters
        ----------
        ax : pyplot axis
            Axis to plot on
        chan_mod : ChanMod structure
            Channel model.
        dvec : (nlink,ndim) array
                vector from cell to UAV
        nlos_ang : (nlink,npaths_max,nangle) array
                Angles of each path in each link.  
                The angles are in degrees
        nlos_pl : (nlink,npaths_max) array 
                Path losses of each path in each link.
                A value of pl_max indicates no path
        iang: integer from 0 to DataFormat.nangle-1
            Index of the angle to be plotted
        np_plot:  integer
            Number of paths whose angles are to be plotted
        """
        # Get the distances
        
#        dvec = self.training_data['dvec'][:,:10]
        if not generated:
            dist = np.sqrt(np.sum(dvec**2,axis=1))
        dist_plot = np.tile(dist[:,None],(1,10))
        dist_plot = dist_plot.ravel()
        
        # Transform the angles.  The transformations compute the
        # relative angles and scales them by 180
#        nlos_ang = self.training_data['nlos_ang'][:,:10]
#        nlos_pl = self.training_data['nlos_pl'][:,:10]
        if not generated:
            ang_tr = self.transform_ang(dvec, nlos_ang, nlos_pl)
        
        ang_rel = ang_tr[:,iang*10:(iang+1)*10]*180
        ang_rel = ang_rel.ravel()
        
        # Find valid paths
        pl_tgt = np.minimum(nlos_pl[:,0]+pl_tol, -150)
        Ivalid = (nlos_pl < pl_tgt[:,None])
        Ivalid = np.where(Ivalid.ravel())[0]
        
        # Get the valid distances and relative angles
        ang_rel = ang_rel[Ivalid]
        dist_plot = dist_plot[Ivalid]      
        
        # Set the angle and distance range for the histogram
        drange = [0,dmax]
        if iang==AngleFormat.aoa_phi_ind or iang==AngleFormat.aod_phi_ind:
            ang_range = [-180,180]
        elif iang==AngleFormat.aoa_theta_ind or iang==AngleFormat.aod_theta_ind:
            ang_range = [-90,90]
        else:
            raise ValueError('Invalid angle index')
        
        # Compute the emperical conditional probability
        H0, dedges, ang_edges = np.histogram2d(dist_plot,ang_rel,bins=[10,40],\
                                               range=[drange,ang_range])       
        Hsum = np.sum(H0,axis=1)
        H0 = H0 / Hsum[:,None]
        
        # Plot the log probability.
        # We plot the log proability since the probability in linear
        # scale is difficult to view
        log_prob = np.log10(np.maximum(0.01,H0.T))
        im = ax.imshow(log_prob, extent=[np.min(dedges),np.max(dedges),\
                   np.min(ang_edges),np.max(ang_edges)], aspect='auto')   
        return im
    
    def eval_angle_dist(self):
        """
        Plot the angular distributions
        """    
        plt.rcParams.update({'font.size': 12})
        
        ang_str = ['$\phi_k^{tx}$', '$\\theta_k^{tx}$', \
                   '$\phi_k^{rx}$', '$\\theta_k^{rx}$']
        
        
        fig, ax = plt.subplots(AngleFormat.nangle, 2, figsize=(5,10))
        for iang in range(AngleFormat.nangle):
            
            for j in range(2):
                if j == 0:
                    data = self.training_data
                else:
                    data = self.generated_data
                                
                axi = ax[iang,j]
                im = self.plot_ang_dist(axi,data['dvec'],data['nlos_ang'],\
                              data['nlos_pl'],iang,dmax=1500)
                    
                if iang < 3:
                    axi.set_xticks([])
                else:
                    axi.set_xlabel('Dist (m)')
                if j == 1:
                    axi.set_yticks([])
                    title_str = ang_str[iang] + ' Model'   
                else:
                    title_str = ang_str[iang] + ' Data'   
                axi.set_title(title_str)
        fig.tight_layout()
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        
        