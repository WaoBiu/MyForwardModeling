#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class Model():
    
    def __init__(self, pmodel, vpmodel, vsmodel, xlen, zlen, dx, dz, nx, nz):
        if xlen==None or zlen==None:
            xlen = dx * nx
            zlen = dz * nz
        elif dx==None or dz==None:
            dx = xlen / dx
            dz = zlen / dz
        elif nz==None or nx==None:
            nx = int(xlen / dx) + 1
            nz = int(zlen / dz) + 1
            
        self._pmodel = pmodel
        self._vpmodel = vpmodel
        self._vsmodel = vsmodel
        self._xlength = xlen
        self._zlength = zlen
        self._npx = nx
        self._npz = nz
        self._xspacing = dx
        self._zspacing = dz
        
    def show(self, which="vpmodel"):
        if which=="vpmodel":
            model = self._vpmodel
        elif which=="vsmodel":
            model = self._vsmodel
        else:
            model = self._pmodel
            
        print(model)
        
        plt.figure()
        plt.imshow(model.T, cmap=plt.cm.coolwarm)
        plt.colorbar(shrink=0.9)
        
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        
        plt.xticks(np.linspace(0, self._npx, 5),np.linspace(0, self._xlength, 5))
        plt.yticks(np.linspace(0, self._npz, 5),np.linspace(0, self._zlength, 5))
        
        plt.show()