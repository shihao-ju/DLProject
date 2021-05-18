# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:58:12 2021

@author: willi
"""

class layer_3ggpUAS_LOS(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate parameters to estimate and assign 
        them as member parameters.
        """
        super().__init__()
        self.d10 = torch.nn.Parameter(torch.randn(()))
        self.p10 = torch.nn.Parameter(torch.randn(()))
        self.a1 = torch.nn.Parameter(torch.randn(()))
        self.b1 = torch.nn.Parameter(torch.randn(()))
        self.a2 = torch.nn.Parameter(torch.randn(()))
        self.b2 = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept input data and we return an output
        probability per sample
        
        """
        nsamp = np.shape(x)[1]
        output = np.zeros([1,nsamp])
        
        for n in range(nsamp):
            if x[2,n] <= 0:
                d1 = self.d10
                p1 = self.p10
            elif x[2,n] > 0:
                d1 = self.a1*x[0] + self.b1
                p1 = self.a2*x[0] + self.b2

            plos = min(1, d1/x[1] + np.exp(-d2d/d1)*(1-d1/x[1]))
            
            output[n] = np.log(plos/(1-plos))   

        return output   