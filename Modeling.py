#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from math import pi, exp, log10, sqrt, log
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# In[2]:
class FirstOlderForward:
    
    # constant parameters used in computing
    _c1 =  1.211243
    _c2 = -0.08972168
    _c3 =  0.001384277
    _c4 = -0.00176566
    _c5 =  0.0001186795
    
    # model parameters
    _pmodel = None
    _vpmodel = None
    _vsmodel = None
    _c11 = None   # = c33 = λ+ 2μ
    _c13 = None   # = λ
    _c44 = None   # = μ
    _vx = None
    _vx = None
    _dt = 0
    _dx = 0
    _dz = 0
    _nt = 0
    _nx = 0
    _nz = 0
    _tmax = 0
    _xmax = 0
    _zmax = 0
    
    def __init__(self, pmodel=None, vpmodel=None, vsmodel=None, tmax=1., xmax=1000., zmax=800., nt=1000, nx=100, nz=80):

        self._pmodel = pmodel
        self._vpmodel = vpmodel
        self._vsmodel = vsmodel
        
        # setting other parameters     
        self._tmax = tmax
        self._xmax = xmax
        self._zmax = zmax
        self._nt = nt
        self._nx = nx
        self._nz = nz
        
        # computing some parameters
        self._dt = tmax/nt
        self._dx = xmax/nx
        self._dz = zmax/nz
        
        self._c11 = self._pmodel * np.power(self._vpmodel, 2)
        self._c44 = self._pmodel * np.power(self._vsmodel, 2)
        self._c13 = self._c11 - 2 * self._c44

        # initialize the u
        self._vx = np.zeros((nx, nz, nt), dtype=float)
        self._vz = np.zeros((nx, nz, nt), dtype=float)

    # forward modeling operator with precion of O(4,10)
    def o4xFM(self, wavelet, wavalet_position, wavalet_direction='z'):
        
        # setting time array
        t_array = np.arange(0, self._tmax, self._dt)
        
        # initialize parameter used in process
        u = np.zeros((self._nx, self._nz), dtype=float)
        v = np.zeros((self._nx, self._nz), dtype=float)
        r = np.zeros((self._nx, self._nz), dtype=float)
        t = np.zeros((self._nx, self._nz), dtype=float)
        h = np.zeros((self._nx, self._nz), dtype=float)
        
        # start to compute
        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute
                u = self.o4xComputeVx(u, v, r, t, h)
                v = self.o4xComputeVz(u, v, r, t, h)
                r = self.o4xComputeTauxx(u ,v, r, t, h)
                t = self.o4xComputeTauzz(u ,v, r, t, h)
                h = self.o4xComputeTauxz(u ,v, r, t, h)
                
                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[0], wavalet_position[1]] += wavelet[tk]
                    else:
                        v[wavalet_position[0], wavalet_position[1]] += wavelet[tk] 
                
            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v
                
            if (np.max(u)>20):
                print("divergent! Please reset gird spacing or/and time step length.")
                return 
      
    # function to oompute the velocity in x axis of the next time step

    def o4xComputeVx(self, u, v, r, t, h):
        output = np.zeros((self._nx, self._nz), dtype=float)
        
        # compute the 1st item
        kernal = np.zeros((11,11), dtype=float)
        kernal[:,5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]
        one = self._dt / (self._dx * self._pmodel) * convolve(r, kernal, mode='same')
        
        # compute the 2nd item
        kernal = kernal.T
        two = self._dt / (self._dz * self._pmodel) * convolve(h, kernal, mode='same')
        
        # compute the 3rd item
        kernal = np.zeros((5, 5), dtype=float)
        kernal[:,2] = [0, 1, -3, 3, -1]
        three = self._dt**3 * self._c11 / (24 * self._pmodel**2 * self._dx**3) * convolve(r, kernal, mode='same')
        
        # compute the 4th item
        kernal = np.array([[0, 1, -1],[0, -2, 2],[0, 1, -1]])
        four = self._dt**3 * (self._c44 + self._c11 + self._c13)                 / (24 * self._pmodel**2 * self._dz * self._dx**2) * convolve(h, kernal, mode='same')
        
        # compute the 5th item
        kernal = np.array([[0, 0, 0],[1, -2, 1],[-1, 2, -1]])
        five = self._dt**3 * (self._c44 + self._c13)                 / (24 * self._pmodel**2 * self._dz**2 * self._dx) * convolve(t, kernal, mode='same')      
        
        # compute the 6th item
        kernal = np.array([[0, 0, 0],[1, -2, 1],[-1, 2, -1]])
        six = self._dt**3 * self._c44                 / (24 * self._pmodel**2 * self._dz**2 * self._dx) * convolve(r, kernal, mode='same')  
        
        # compute the 7th item
        kernal = np.zeros((5, 5), dtype=float)
        kernal[2, :] = [0, 1, -3, 3, -1]
        seven = self._dt**3 * self._c44                 / (24 * self._pmodel**2 * self._dz**3) * convolve(h, kernal, mode='same')
        
        # sum all
        output = u + one + two + three + four + five + six + seven
        
        return output    
    
    # function to oompute the velocity in z axis of the next time step

    def o4xComputeVz(self, u, v, r, t, h):
        output = np.zeros((self._nx, self._nz), dtype=float)
        
        # compute the 1st item
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]      
        one = self._dt / (self._dx * self._pmodel) * convolve(h, kernal, mode='same')
        
        # compute the 2nd item
        kernal = kernal.T
        two = self._dt / (self._dz * self._pmodel) * convolve(t, kernal, mode='same')
        
        # compute the 3rd item
        kernal = np.zeros((5, 5), dtype=float)
        kernal[:,2] = [1, -3, 3, -1, 0]
        three = self._dt**3 * self._c44 / (24 * self._pmodel**2 * self._dx**3) * convolve(h, kernal, mode='same')
        
        # compute the 4th item
        kernal = np.array([[1, -2, 1],[-1, 2, -1],[0, 0, 0]])
        four = self._dt**3 * (self._c44 + self._c11 + self._c13)                 / (24 * self._pmodel**2 * self._dx * self._dz**2) * convolve(h, kernal, mode='same')
        
        # compute the 5th item
        kernal = np.array([[1, -1, 0],[-2, 2, 0],[1, -1, 0]])
        five = self._dt**3 * (self._c44 + self._c13)                 / (24 * self._pmodel**2 * self._dx**2 * self._dz) * convolve(r, kernal, mode='same')      
        
        # compute the 6th item
        kernal = np.array([[1, -1, 0],[-2, 2, 0],[1, -1, 0]])
        six = self._dt**3 * self._c44                 / (24 * self._pmodel**2 * self._dx**2 * self._dz) * convolve(t, kernal, mode='same')
        
        # compute the 7th item
        kernal = np.zeros((5, 5), dtype=float)
        kernal[2,:] = [1, -3, 3, -1, 0]
        seven = self._dt**3 * self._c11                 / (24 * self._pmodel**2 * self._dz**3) * convolve(t, kernal, mode='same')
        
        # sum all
        output = v + one + two + three + four + five + six + seven
        
        return output

    # function to oompute the stress on xx
    def o4xComputeTauxx(self, u, v, r, t, h):
        output = np.zeros((self._nx, self._nz), dtype=float)
        
        # compute the 1st item
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]      
        one = self._dt * self._c11 / self._dx * convolve(u, kernal, mode='same')
        
        # compute the 2nd item
        kernal = np.zeros((11,11),dtype=float)
        kernal[5,:] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]   
        two = self._dt * self._c13 / self._dz * convolve(v, kernal, mode='same')
        
        # computer the 3rd item
        kernal = np.zeros((5,5), dtype=float)
        kernal[:,2] = [1, -3, 3, -1, 0]
        three = self._dt**3 * self._c11**2 / (24 * self._pmodel * self._dx**3) * convolve(u, kernal, mode='same')
        
        # computer the 4th item
        kernal = np.array([[0, 1, -1],[0, -2, 2],[0, 1, -1]])
        four = self._dt**3 * (self._c11*self._c13 + self._c11*self._c44 + self._c13*self._c44)                 / (24 * self._pmodel * self._dx**2 * self._dz) * convolve(v, kernal, mode='same')  
        
        # computer the 5th item
        kernal = np.array([[1, -2, 1],[-1, 2, -1],[0, 0, 0]])
        five = self._dt**3 * (self._c13**2 + self._c11*self._c44 + self._c13*self._c44)                 / (24 * self._pmodel * self._dz**2 * self._dx) * convolve(u, kernal, mode='same')  

        # computer the 6th item
        kernal = np.zeros((5,5), dtype=float)
        kernal[2, :] = [0, 1, -3, 3, -1]
        six = self._dt**3 * self._c11* self._c13 / (24 * self._pmodel * self._dz**3) * convolve(v, kernal, mode='same')
        
        # sum all
        output = r + one + two + three + four + five + six
        
        return output
    
    # function to oompute the stress on zz
    def o4xComputeTauzz(self, u, v, r, t, h):
        output = np.zeros((self._nx, self._nz), dtype=float)
        
        # compute the 1st item
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]     
        one = self._dt * self._c13 / self._dx * convolve(u, kernal, mode='same')
        
        # compute the 2nd item
        kernal = np.zeros((11,11),dtype=float)
        kernal[5,:] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]            
        two= self._dt * self._c11 / self._dz * convolve(v, kernal, mode='same')
        
        # computer the 3rd item
        kernal = np.zeros((5,5), dtype=float)
        kernal[:,2] = [1, -3, 3, -1, 0]
        three = self._dt**3 * self._c11 * self._c13 / (24 * self._pmodel * self._dx**3) * convolve(u, kernal, mode='same')
        
        # computer the 4th item
        kernal = np.array([[0, 1, -1],[0, -2, 2],[0, 1, -1]])
        four = self._dt**3 * (self._c13**2 + self._c13*self._c44 + self._c11*self._c44)                 / (24 * self._pmodel * self._dx**2 * self._dz) * convolve(v, kernal, mode='same')  
        
        # computer the 5th item
        kernal = np.array([[1, -2, 1],[-1, 2, -1],[0, 0, 0]])
        five = self._dt**3 * (self._c13 * self._c11 + self._c13 * self._c44 + self._c11 * self._c44)                 / (24 * self._pmodel * self._dz**2 * self._dx) * convolve(u, kernal, mode='same')  

        # computer the 6th item
        kernal = np.zeros((5,5), dtype=float)
        kernal[2, :] = [0, 1, -3, 3, -1]
        six = self._dt**3 * self._c11**2 / (24 * self._pmodel * self._dz**3) * convolve(v, kernal, mode='same')
        
        # sum all
        output = t + one + two + three + four + five + six
        
        return output
    
    # function to oompute the stress on xz
    def o4xComputeTauxz(self, u, v, r, t, h):
        output = np.zeros((self._nx, self._nz), dtype=float)
        
        # compute the 1st item
        kernal = np.zeros((11, 11),dtype=float)
        kernal[:,5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]     
        one = self._dt * self._c44 / self._dx * convolve(v, kernal, mode='same')
        
        # compute the 2nd item
        kernal = np.zeros((11,11),dtype=float)
        kernal[5,:] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]   
        two = self._dt * self._c44 / self._dz * convolve(u, kernal, mode='same')
        
        # computer the 3rd item
        kernal = np.zeros((5,5), dtype=float)
        kernal[:,2] = [0, 1, -3, 3, -1]
        three = self._dt**3 * self._c44**2 / (24 * self._pmodel * self._dx**3) * convolve(v, kernal, mode='same')
        
        # computer the 4th item
        kernal = np.array([[1, -1, 0],[-2, 2, 0],[1, -1, 0]])
        four = self._dt**3 * (self._c44**2 + self._c11*self._c44 + self._c13*self._c44)                 / (24 * self._pmodel * self._dx**2 * self._dz) * convolve(u, kernal, mode='same')  
        
        # computer the 5th item
        kernal = np.array([[0,0,0],[1, -2, 1],[-1, 2, -1]])
        five = self._dt**3 * self._c44 * (self._c13 + self._c44 + self._c11)                 / (24 * self._pmodel * self._dz**2 * self._dx) * convolve(v, kernal, mode='same')  

        # computer the 6th item
        kernal = np.zeros((5,5), dtype=float)
        kernal[2, :] = [1, -3, 3, -1, 0]
        six = self._dt**3 * self._c44**2 / (24 * self._pmodel * self._dz**3) * convolve(u, kernal, mode='same')
        
        # sum all
        output = h + one + two + three + four + five + six
        
        return output

    def o2anyFM(self, cal_par, wavelet, wavalet_position, wavalet_direction='z'):
    # cal_par ： the parameters using to calculate the derivative

        # setting time array
        t_array = np.arange(0, self._tmax, self._dt)

        # initialize parameter used in process
        u = np.zeros((self._nx, self._nz), dtype=float)
        v = np.zeros((self._nx, self._nz), dtype=float)
        r = np.zeros((self._nx, self._nz), dtype=float)
        t = np.zeros((self._nx, self._nz), dtype=float)
        h = np.zeros((self._nx, self._nz), dtype=float)

        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute

                u_x = self.o2any_cal_u_x(u, cal_par)
                u_z = self.o2any_cal_u_z(u, cal_par)
                v_x = self.o2any_cal_v_x(v, cal_par)
                v_z = self.o2any_cal_v_z(v, cal_par)

                r = self.ComputeTauxx(r, u_x, v_z)
                t = self.ComputeTauzz(t, u_x, v_z)
                h = self.ComputeTauxz(h, v_x, u_z)

                r_x = self.o2any_cal_r_x(r, cal_par)
                t_z = self.o2any_cal_t_z(t, cal_par)
                h_x = self.o2any_cal_h_x(h, cal_par)
                h_z = self.o2any_cal_h_z(h, cal_par)

                u = self.ComputeVx(u, r_x, h_z)
                v = self.ComputeVz(v, t_z, h_x)

                if tk < len(wavelet):  # source is active
                    if wavalet_direction == 'x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:, :, tk] = u
            self._vz[:, :, tk] = v

            if (np.max(u) > 20):
                print("divergent! Please reset gird spacing or/and time step length.")
                return
    
    def o22FM(self, wavelet, wavalet_position, wavalet_direction='z'): 
        
        # setting time array
        t_array = np.arange(0, self._tmax, self._dt)
        
        # initialize parameter used in process
        u = np.zeros((self._nx, self._nz), dtype=float)
        v = np.zeros((self._nx, self._nz), dtype=float)
        r = np.zeros((self._nx, self._nz), dtype=float)
        t = np.zeros((self._nx, self._nz), dtype=float)
        h = np.zeros((self._nx, self._nz), dtype=float)
        
        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute
                
                u_x = self.o22_cal_u_x(u)
                u_z = self.o22_cal_u_z(u)
                v_x = self.o22_cal_v_x(v)
                v_z = self.o22_cal_v_z(v)
                
                r = self.ComputeTauxx(r, u_x ,v_z)
                t = self.ComputeTauzz(t, u_x ,v_z)
                h = self.ComputeTauxz(h ,v_x, u_z)

                r_x = self.o22_cal_r_x(r)
                t_z = self.o22_cal_t_z(t)
                h_x = self.o22_cal_h_x(h)
                h_z = self.o22_cal_h_z(h)

                u = self.ComputeVx(u, r_x, h_z)
                v = self.ComputeVz(v, t_z, h_x)

                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk] 
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v
            
            if (np.max(u)>20):
                print("divergent! Please reset gird spacing or/and time step length.")
                return
    
    def o24FM(self, wavelet, wavalet_position, wavalet_direction='z'): 
        
        # setting time array
        t_array = np.arange(0, self._tmax, self._dt)
        
        # initialize parameter used in process
        u = np.zeros((self._nx, self._nz), dtype=float)
        v = np.zeros((self._nx, self._nz), dtype=float)
        r = np.zeros((self._nx, self._nz), dtype=float)
        t = np.zeros((self._nx, self._nz), dtype=float)
        h = np.zeros((self._nx, self._nz), dtype=float)
        
        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute
                
                u_x = self.o24_cal_u_x(u)
                u_z = self.o24_cal_u_z(u)
                v_x = self.o24_cal_v_x(v)
                v_z = self.o24_cal_v_z(v)
                
                r = self.ComputeTauxx(r, u_x ,v_z)
                t = self.ComputeTauzz(t, u_x ,v_z)
                h = self.ComputeTauxz(h ,v_x, u_z)

                r_x = self.o24_cal_r_x(r)
                t_z = self.o24_cal_t_z(t)
                h_x = self.o24_cal_h_x(h)
                h_z = self.o24_cal_h_z(h)

                u = self.ComputeVx(u, r_x, h_z)
                v = self.ComputeVz(v, t_z, h_x)
                
                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk] 
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v
            
            if (np.max(u)>20):
                print("divergent! Please reset gird spacing or/and time step length.")
                return
    # some error
#     def o26FM(self, wavelet, wavalet_position, wavalet_direction='z'):    
        
#         # setting time array
#         t_array = np.arange(0, self._tmax, self._dt)
        
#         # initialize parameter used in process
#         u = np.zeros((self._nx, self._nz), dtype=float)
#         v = np.zeros((self._nx, self._nz), dtype=float)
#         r = np.zeros((self._nx, self._nz), dtype=float)
#         t = np.zeros((self._nx, self._nz), dtype=float)
#         h = np.zeros((self._nx, self._nz), dtype=float)
#         for tk, tt in enumerate(t_array):
#             if tk >= 1:  # the first step needs not to compute
#                 u = self.o26ComputeVx(u, v, r, t, h)
#                 v = self.o26ComputeVz(u, v, r, t, h)
#                 r = self.o26ComputeTauxx(u ,v, r, t, h)
#                 t = self.o26ComputeTauzz(u ,v, r, t, h)
#                 h = self.o26ComputeTauxz(u ,v, r, t, h)
                
#                 if tk < len(wavelet):  # source is active
#                     if wavalet_direction=='x':
#                         u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
#                     else:
#                         v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

#             self._vx[:,:,tk] = u
#             self._vz[:,:,tk] = v
            
#             if tk % 100 == 99:
#                 print("The " + str(tk) + " times result of v_x near the source position")
#                 print(u[wavalet_position[1]-3:wavalet_position[0]+4, wavalet_position[1]-3:wavalet_position[0]+4])
            
#     def o26ComputeVx(self, u, v, r, t, h):
              
#         output = np.zeros((self._nx, self._nz), dtype=float)
        
#         kernal = np.zeros((7,7), dtype=float)
#         kernal[:,3] = [0, 0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875]
        
#         one = self._dt / (self._pmodel * self._dx) * convolve(r, kernal, mode='same')
        
#         kernal = kernal.T
#         two = self._dt / (self._pmodel * self._dz) * convolve(h, kernal, mode='same')  
#         output = u + one + two
        
#         return output
    
#     def o26ComputeVz(self, u, v, r, t, h):
         
#         output = np.zeros((self._nx, self._nz), dtype=float)
        
#         kernal = np.zeros((7,7),dtype=float)
#         kernal[:,3] = [0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875, 0]
#         one = self._dt / (self._pmodel * self._dx) * convolve(h, kernal, mode='same')
        
#         kernal = kernal.T
#         two = self._dt / (self._pmodel * self._dz) * convolve(t, kernal, mode='same')
        
#         output = v + one + two

#         return output
    
#     def o26ComputeTauxx(self, u, v, r, t, h):
        
#         output = np.zeros((self._nx, self._nz), dtype=float)

#         kernal = np.zeros((7,7),dtype=float)
#         kernal[:,3] = [0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875, 0]
#         one = (self._dt * self._c11) / (self._dx) * convolve(u, kernal, mode='same')
        
#         kernal = np.zeros((7,7),dtype=float)
#         kernal[3,:] = [0, 0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875]
#         two = (self._dt * self._c13) / (self._dz) * convolve(v, kernal, mode='same')
        
#         output = r + one + two
        
#         return output
    
#     def o26ComputeTauzz(self, u, v, r, t, h):
        
#         output = np.zeros((self._nx, self._nz), dtype=float)
        
#         kernal = np.zeros((7,7),dtype=float)
#         kernal[:,3] = [0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875, 0]
#         two = (self._dt * self._c13) / (self._dx) * convolve(u, kernal, mode='same')
        
#         kernal = np.zeros((7,7),dtype=float)
#         kernal[3,:] = [0, 0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875]
#         one = (self._dt * self._c11) / (self._dz) * convolve(v, kernal, mode='same')
        
#         output = t + one + two     
        
#         return output
    
#     def o26ComputeTauxz(self, u, v, r, t, h):
        
#         output = np.zeros((self._nx, self._nz), dtype=float)
        
#         kernal = np.zeros((7,7),dtype=float)
#         kernal[3,:] = [0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875, 0]
#         one = (self._dt * self._c44) / (self._dz) * convolve(u, kernal, mode='same')
        
#         kernal = np.zeros((7,7),dtype=float)
#         kernal[:,3] = [0, 0.0046875, -0.06510416, 1.171875, -1.171875, 0.06510416, -0.0046875]
#         two = (self._dt * self._c44) / (self._dx) * convolve(v, kernal, mode='same')
           
#         output = h + one + two

#         return output

    def o2xFM(self, wavelet, wavalet_position, wavalet_direction='z'): 
        
        # setting time array
        t_array = np.arange(0, self._tmax, self._dt)
        
        # initialize parameter used in process
        u = np.zeros((self._nx, self._nz), dtype=float)
        v = np.zeros((self._nx, self._nz), dtype=float)
        r = np.zeros((self._nx, self._nz), dtype=float)
        t = np.zeros((self._nx, self._nz), dtype=float)
        h = np.zeros((self._nx, self._nz), dtype=float)
        
        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute
                
                u_x = self.o2x_cal_u_x(u)
                u_z = self.o2x_cal_u_z(u)
                v_x = self.o2x_cal_v_x(v)
                v_z = self.o2x_cal_v_z(v)
                
                r = self.ComputeTauxx(r, u_x ,v_z)
                t = self.ComputeTauzz(t, u_x ,v_z)
                h = self.ComputeTauxz(h ,v_x, u_z)

                r_x = self.o2x_cal_r_x(r)
                t_z = self.o2x_cal_t_z(t)
                h_x = self.o2x_cal_h_x(h)
                h_z = self.o2x_cal_h_z(h)

                u = self.ComputeVx(u, r_x, h_z)
                v = self.ComputeVz(v, t_z, h_x)
                
                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk] 
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v
            
            if (np.max(u)>20):
                print("divergent! Please reset gird spacing or/and time step length.")
                return

    # Derivation function
    def o22_cal_r_x(self, r):
        kernal = np.array([[0,0,0],[0,1,0],[0,-1,0]])
        return convolve(r, kernal, mode='same') / self._dx
    
    def o22_cal_h_z(self, h):
        kernal = np.array([[0,0,0],[0,1,0],[0,-1,0]])
        kernal = kernal.T
        return convolve(h, kernal, mode='same') / self._dz
    
    def o22_cal_h_x(self, h):
        kernal = np.array([[0,1,0],[0,-1,0],[0,0,0]])
        return convolve(h, kernal, mode='same') / self._dx
    
    def o22_cal_t_z(self, t):
        kernal = np.array([[0,1,0],[0,-1,0],[0,0,0]])
        kernal = kernal.T
        return convolve(t, kernal, mode='same') / self._dz
    
    def o22_cal_u_x(self, u):
        kernal = np.array([[0,1,0],[0,-1,0],[0,0,0]])
        return convolve(u, kernal, mode='same') / self._dx

    def o22_cal_u_z(self, u):
        kernal = np.array([[0,0,0],[1,-1,0],[0,0,0]])
        return convolve(u, kernal, mode='same') / self._dz

    def o22_cal_v_x(self, v):
        kernal = np.array([[0,0,0],[0,1,0],[0,-1,0]])
        return convolve(v, kernal, mode='same') / self._dx
    
    def o22_cal_v_z(self, v):
        kernal = np.array([[0,0,0],[0,1,-1],[0,0,0]])
        return convolve(v, kernal, mode='same') / self._dz
    
    def o24_cal_r_x(self, r):
        kernal = np.array([[0,0,0,0,0],[0,0,-0.04166667,0,0],[0,0,1.125,0,0],[0,0,-1.125,0,0],[0,0,0.04166667,0,0]])
        return convolve(r, kernal, mode='same') / self._dx
    
    def o24_cal_h_z(self, h):
        kernal = np.array([[0,0,0,0,0],[0,0,-0.04166667,0,0],[0,0,1.125,0,0],[0,0,-1.125,0,0],[0,0,0.04166667,0,0]])
        kernal = kernal.T
        return convolve(h, kernal, mode='same') / self._dz   
    
    def o24_cal_h_x(self, h):
        kernal = np.array([[0,0,-0.04166667,0,0],[0,0,1.125,0,0],[0,0,-1.125,0,0],[0,0,0.04166667,0,0],[0,0,0,0,0]])
        return convolve(h, kernal, mode='same') / self._dx
    
    def o24_cal_t_z(self, t):
        kernal = np.array([[0,0,-0.04166667,0,0],[0,0,1.125,0,0],[0,0,-1.125,0,0],[0,0,0.04166667,0,0],[0,0,0,0,0]])
        kernal = kernal.T
        return convolve(t, kernal, mode='same') / self._dz
    
    def o24_cal_u_x(self, u):
        kernal = np.array([[0,0,-0.04166667,0,0],[0,0,1.125,0,0],[0,0,-1.125,0,0],[0,0,0.04166667,0,0],[0,0,0,0,0]])
        return convolve(u, kernal, mode='same') / self._dx

    def o24_cal_u_z(self, u):
        kernal = np.array([[0,0,0,0,0],[0,0,0,0,0],[-0.04166667,1.125,-1.125,0.04166667,0],[0,0,0,0,0],[0,0,0,0,0]])
        return convolve(u, kernal, mode='same') / self._dz

    def o24_cal_v_x(self, v):
        kernal = np.array([[0,0,0,0,0],[0,0,-0.04166667,0,0],[0,0,1.125,0,0],[0,0,-1.125,0,0],[0,0,0.04166667,0,0]])
        return convolve(v, kernal, mode='same') / self._dx
    
    def o24_cal_v_z(self, v):
        kernal = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-0.04166667,1.125,-1.125,0.04166667],[0,0,0,0,0],[0,0,0,0,0]])
        return convolve(v, kernal, mode='same') / self._dz
    
    def o2x_cal_r_x(self, r):
        kernal = np.zeros((11,11), dtype=float)
        kernal[:,5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]
        return convolve(r, kernal, mode='same') / self._dx
    
    def o2x_cal_h_z(self, h):
        kernal = np.zeros((11,11), dtype=float)
        kernal[:,5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]
        kernal = kernal.T
        return convolve(h, kernal, mode='same') / self._dz
    
    def o2x_cal_h_x(self, h):
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]
        return convolve(h, kernal, mode='same') / self._dx
    
    def o2x_cal_t_z(self, t):
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]
        kernal = kernal.T
        return convolve(t, kernal, mode='same') / self._dz
    
    def o2x_cal_u_x(self, u):
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]
        return convolve(u, kernal, mode='same') / self._dx

    def o2x_cal_u_z(self, u):
        kernal = np.zeros((11,11),dtype=float)
        kernal[5,:] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]
        return convolve(u, kernal, mode='same') / self._dz

    def o2x_cal_v_x(self, v):
        kernal = np.zeros((11, 11),dtype=float)
        kernal[:,5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]
        return convolve(v, kernal, mode='same') / self._dx
    
    def o2x_cal_v_z(self, v):
        kernal = np.zeros((11,11),dtype=float)
        kernal[5,:] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]
        return convolve(v, kernal, mode='same') / self._dz

    def o2any_cal_r_x(self, r, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[1:array_mid+1, array_mid] = cal_par[::-1]
        kernal[array_mid+1:, array_mid] = cal_par * -1
        # kernal[:, 5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5]
        return convolve(r, kernal, mode='same') / self._dx

    def o2any_cal_h_z(self, h, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[1:array_mid+1, array_mid] = cal_par[::-1]
        kernal[array_mid+1:, array_mid] = cal_par * -1
        # kernal = np.zeros((11, 11), dtype=float)
        # kernal[:, 5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5]
        kernal = kernal.T
        return convolve(h, kernal, mode='same') / self._dz

    def o2any_cal_h_x(self, h, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[0:array_mid, array_mid] = cal_par[::-1]
        kernal[array_mid:-1, array_mid] = cal_par * -1
        # kernal = np.zeros((11, 11), dtype=float)
        # kernal[:, 5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5, 0]
        return convolve(h, kernal, mode='same') / self._dx

    def o2any_cal_t_z(self, t, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[0:array_mid, array_mid] = cal_par[::-1]
        kernal[array_mid:-1, array_mid] = cal_par * -1
        # kernal = np.zeros((11, 11), dtype=float)
        # kernal[:, 5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5, 0]
        kernal = kernal.T
        return convolve(t, kernal, mode='same') / self._dz

    def o2any_cal_u_x(self, u, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[0:array_mid, array_mid] = cal_par[::-1]
        kernal[array_mid:-1, array_mid] = cal_par * -1
        # kernal[:, 5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5, 0]
        return convolve(u, kernal, mode='same') / self._dx

    def o2any_cal_u_z(self, u, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[array_mid, 0:array_mid] = cal_par[::-1]
        kernal[array_mid, array_mid:-1] = cal_par * -1
        # kernal[5, :] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5, 0]
        return convolve(u, kernal, mode='same') / self._dz

    def o2any_cal_v_x(self, v, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[1:array_mid+1, array_mid] = cal_par[::-1]
        kernal[array_mid+1:, array_mid] = cal_par * -1
        # kernal[:, 5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5]
        return convolve(v, kernal, mode='same') / self._dx

    def o2any_cal_v_z(self, v, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[array_mid, 1:array_mid+1] = cal_par[::-1]
        kernal[array_mid, array_mid+1:] = cal_par * -1
        # kernal[5, :] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5]
        return convolve(v, kernal, mode='same') / self._dz

    def ComputeVx(self, u, r_x, h_z):
        one = self._dt / self._pmodel * r_x
        two = self._dt / self._pmodel * h_z
        return u + one + two

    def ComputeVz(self, v, t_z, h_x):
        one = self._dt / self._pmodel * h_x
        two = self._dt / self._pmodel * t_z
        return v + one + two

    def ComputeTauxx(self, r, u_x, v_z):
        one = (self._dt * self._c11) * u_x
        two = (self._dt * self._c13) * v_z
        return r + one + two

    def ComputeTauzz(self, t, u_x, v_z):
        one = (self._dt * self._c11) * v_z
        two = (self._dt * self._c13) * u_x
        return t + one + two

    def ComputeTauxz(self, h, v_x, u_z):
        one = (self._dt * self._c44) * u_z
        two = (self._dt * self._c44) * v_x
        return h + one + two
    
    # painting function
    def DrawModel(self):
        if self._nx > self._nz:
            plt.figure(figsize=(20, 4))
        elif self._nx == self._nz:
            plt.figure(figsize=(20, 6))
        else:
            plt.figure(figsize=(20, 8))
        
        ax1 = plt.subplot(1,3,1)
        plt.imshow(self._pmodel.T, cmap=plt.cm.cool, aspect='auto')
        plt.title("density model")
        plt.colorbar(shrink=0.8)
        
        plt.subplot(1,3,2)
        plt.imshow(self._vpmodel.T, cmap=plt.cm.cool, aspect='auto')
        plt.title("P wave velocity model")
        plt.colorbar(shrink=0.8)

        plt.subplot(1,3,3)
        plt.imshow(self._vsmodel.T, cmap=plt.cm.cool, aspect='auto')
        plt.title("S wave velocitydensity model")
        plt.colorbar(shrink=0.8)
        
    def RangeInOne(self, array):
        _max = max(np.max(array), abs(np.min(array)))
        return array / _max
        
    def DrawXWaveField(self, iterations_to_show):
        fgr, axs = plt.subplots(1,len(iterations_to_show), figsize = (18,10))
        for j, ax in enumerate(axs):
            ax.imshow(self.RangeInOne(self._vx[:, :, iterations_to_show[j]].T),                       cmap = plt.cm.coolwarm, vmin = -1, vmax = 1, interpolation='bilinear')
            ax.annotate("t = {0} ms".format(iterations_to_show[j] * self._dt), xy=(0.05, 0.05), xycoords="axes fraction")
        plt.show()
        
    def DrawZWaveField(self, iterations_to_show):
        fgr, axs = plt.subplots(1,len(iterations_to_show), figsize = (18,10))
        for j, ax in enumerate(axs):
            ax.imshow(self.RangeInOne(self._vz[:, :, iterations_to_show[j]].T),                       cmap = plt.cm.coolwarm, vmin = -1, vmax = 1, interpolation='bilinear')
            ax.annotate("t = {0} ms".format(iterations_to_show[j]), xy=(0.05, 0.05), xycoords="axes fraction")
        plt.show()
        
    def DrawXRecord(self, amp = 1.0):
        surface_record_no_bc = self._vx[:,0,:]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap = 'gray', vmin = -1, vmax = 1, interpolation='bilinear', aspect='auto')
        plt.title("Record")
        plt.show()

    def DrawZRecord(self, amp = 1.0):
        surface_record_no_bc = self._vz[:,0,:]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap = 'gray', vmin = -1, vmax = 1, interpolation='bilinear', aspect='auto')
        plt.title("Record")
        plt.show()

# In[3]:
class FirstOlder_SPML(FirstOlderForward):
    
    def __init__(self, pmodel, vpmodel, vsmodel, tmax, xmax, zmax, nt, nx, nz, pml_len):
        
        super().__init__(pmodel, vpmodel, vsmodel, tmax, xmax, zmax, nt, nx, nz)
        self._pml_len = pml_len
        
        dmax = 10 * np.max(self._vpmodel) / pml_len #4.5
        # print(dmax)
        xspacing = np.linspace(0,nx,nx)
        zspacing = np.linspace(0,nz,nz)
        zmesh, xmesh = np.meshgrid(zspacing, xspacing)
        
        self._pml_x = np.where(xmesh<pml_len, dmax * np.power((pml_len-xmesh)/pml_len,4),                                 np.where(nx-xmesh<pml_len, dmax * np.power((xmesh-nx+pml_len)/pml_len,4), 0))
        self._pml_z = np.where(zmesh<pml_len, dmax * np.power((pml_len-zmesh)/pml_len,4),                                 np.where(nz-zmesh<pml_len, dmax * np.power((zmesh-nz+pml_len)/pml_len,4), 0))
    
    def o24FM(self, wavelet, wavalet_position, wavalet_direction='z'):
        # u,v,r,t,h,_1,_2,_sum
        nx = self._nx
        nz = self._nz
        t_array = np.arange(0, self._tmax, self._dt)
        u = np.zeros((nx,nz), dtype=float)
        v = np.zeros((nx,nz), dtype=float)
        r = np.zeros((nx,nz), dtype=float)
        t = np.zeros((nx,nz), dtype=float)
        h = np.zeros((nx,nz), dtype=float)
        
        u_ver = np.zeros((nx,nz), dtype=float)
        v_ver = np.zeros((nx,nz), dtype=float)
        r_ver = np.zeros((nx,nz), dtype=float)
        t_ver = np.zeros((nx,nz), dtype=float)
        h_ver = np.zeros((nx,nz), dtype=float)
        
        u_par = np.zeros((nx,nz), dtype=float)
        v_par = np.zeros((nx,nz), dtype=float)
        r_par = np.zeros((nx,nz), dtype=float)
        t_par = np.zeros((nx,nz), dtype=float)
        h_par = np.zeros((nx,nz), dtype=float)
        
        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute
                
                u_x = self.o24_cal_u_x(u)
                u_z = self.o24_cal_u_z(u)
                v_x = self.o24_cal_v_x(v)
                v_z = self.o24_cal_v_z(v)
                
                r_ver = self.cal_r_ver(r_ver, u_x)
                r_par = self.cal_r_par(r_par, v_z)
                t_ver = self.cal_t_ver(t_ver, u_x)
                t_par = self.cal_t_par(t_par, v_z)
                h_ver = self.cal_h_ver(h_ver, v_x)
                h_par = self.cal_h_par(h_par, u_z)
                
                r = r_par + r_ver
                r = r_par + r_ver
                t = t_par + t_ver
                t = t_par + t_ver
                h = h_par + h_ver
                h = h_par + h_ver
                
                r_x = self.o24_cal_r_x(r)
                h_x = self.o24_cal_h_x(h)
                h_z = self.o24_cal_h_z(h)
                t_z = self.o24_cal_t_z(t)
                
                u_ver = self.cal_u_ver(u_ver, r_x)
                u_par = self.cal_u_par(u_par, h_z)
                v_ver = self.cal_v_ver(v_ver, h_x)
                v_par = self.cal_v_par(v_par, t_z)
                
                u = u_par + u_ver
                v = v_par + v_ver
                
                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v
            
    def o2xFM(self, wavelet, wavalet_position, wavalet_direction='z'):
        # u,v,r,t,h,_1,_2,_sum
        nx = self._nx
        nz = self._nz
        t_array = np.arange(0, self._tmax, self._dt)
        u = np.zeros((nx,nz), dtype=float)
        v = np.zeros((nx,nz), dtype=float)
        r = np.zeros((nx,nz), dtype=float)
        t = np.zeros((nx,nz), dtype=float)
        h = np.zeros((nx,nz), dtype=float)
        
        u_ver = np.zeros((nx,nz), dtype=float)
        v_ver = np.zeros((nx,nz), dtype=float)
        r_ver = np.zeros((nx,nz), dtype=float)
        t_ver = np.zeros((nx,nz), dtype=float)
        h_ver = np.zeros((nx,nz), dtype=float)
        
        u_par = np.zeros((nx,nz), dtype=float)
        v_par = np.zeros((nx,nz), dtype=float)
        r_par = np.zeros((nx,nz), dtype=float)
        t_par = np.zeros((nx,nz), dtype=float)
        h_par = np.zeros((nx,nz), dtype=float)
        
        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute
                
                u_x = self.o2x_cal_u_x(u)
                u_z = self.o2x_cal_u_z(u)
                v_x = self.o2x_cal_v_x(v)
                v_z = self.o2x_cal_v_z(v)
                
                r_ver = self.cal_r_ver(r_ver, u_x)
                r_par = self.cal_r_par(r_par, v_z)
                t_ver = self.cal_t_ver(t_ver, u_x)
                t_par = self.cal_t_par(t_par, v_z)
                h_ver = self.cal_h_ver(h_ver, v_x)
                h_par = self.cal_h_par(h_par, u_z)
                
                r = r_par + r_ver
                r = r_par + r_ver
                t = t_par + t_ver
                t = t_par + t_ver
                h = h_par + h_ver
                h = h_par + h_ver
                
                r_x = self.o2x_cal_r_x(r)
                h_x = self.o2x_cal_h_x(h)
                h_z = self.o2x_cal_h_z(h)
                t_z = self.o2x_cal_t_z(t)
                
                u_ver = self.cal_u_ver(u_ver, r_x)
                u_par = self.cal_u_par(u_par, h_z)
                v_ver = self.cal_v_ver(v_ver, h_x)
                v_par = self.cal_v_par(v_par, t_z)
                
                u = u_par + u_ver
                v = v_par + v_ver
                
                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v

    def o2anyFM(self, cal_par, wavelet, wavalet_position, wavalet_direction='z'):
        # u,v,r,t,h,_1,_2,_sum
        nx = self._nx
        nz = self._nz
        t_array = np.arange(0, self._tmax, self._dt)
        u = np.zeros((nx, nz), dtype=float)
        v = np.zeros((nx, nz), dtype=float)
        r = np.zeros((nx, nz), dtype=float)
        t = np.zeros((nx, nz), dtype=float)
        h = np.zeros((nx, nz), dtype=float)

        u_ver = np.zeros((nx, nz), dtype=float)
        v_ver = np.zeros((nx, nz), dtype=float)
        r_ver = np.zeros((nx, nz), dtype=float)
        t_ver = np.zeros((nx, nz), dtype=float)
        h_ver = np.zeros((nx, nz), dtype=float)

        u_par = np.zeros((nx, nz), dtype=float)
        v_par = np.zeros((nx, nz), dtype=float)
        r_par = np.zeros((nx, nz), dtype=float)
        t_par = np.zeros((nx, nz), dtype=float)
        h_par = np.zeros((nx, nz), dtype=float)

        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute

                u_x = self.o2any_cal_u_x(u, cal_par)
                u_z = self.o2any_cal_u_z(u, cal_par)
                v_x = self.o2any_cal_v_x(v, cal_par)
                v_z = self.o2any_cal_v_z(v, cal_par)

                r_ver = self.cal_r_ver(r_ver, u_x)
                r_par = self.cal_r_par(r_par, v_z)
                t_ver = self.cal_t_ver(t_ver, u_x)
                t_par = self.cal_t_par(t_par, v_z)
                h_ver = self.cal_h_ver(h_ver, v_x)
                h_par = self.cal_h_par(h_par, u_z)

                r = r_par + r_ver
                r = r_par + r_ver
                t = t_par + t_ver
                t = t_par + t_ver
                h = h_par + h_ver
                h = h_par + h_ver

                r_x = self.o2any_cal_r_x(r, cal_par)
                h_x = self.o2any_cal_h_x(h, cal_par)
                h_z = self.o2any_cal_h_z(h, cal_par)
                t_z = self.o2any_cal_t_z(t, cal_par)

                u_ver = self.cal_u_ver(u_ver, r_x)
                u_par = self.cal_u_par(u_par, h_z)
                v_ver = self.cal_v_ver(v_ver, h_x)
                v_par = self.cal_v_par(v_par, t_z)

                u = u_par + u_ver
                v = v_par + v_ver

                if tk < len(wavelet):  # source is active
                    if wavalet_direction == 'x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:, :, tk] = u
            self._vz[:, :, tk] = v

    def cal_u_ver(self, u_ver, r_x):
        r_x = r_x / self._pmodel
        return (r_x + u_ver * (1 / self._dt - 0.5 * self._pml_x)) / (1 / self._dt + 0.5 *  self._pml_x)
    
    def cal_u_par(self, u_par, h_z):
        h_z = h_z / self._pmodel
        return (h_z + u_par * (1 / self._dt - 0.5 * self._pml_z)) / (1 / self._dt + 0.5 * self._pml_z)
    
    def cal_v_ver(self, v_ver, h_x):
        h_x = h_x / self._pmodel
        return (h_x + v_ver * (1 / self._dt - 0.5 * self._pml_x)) / (1 / self._dt + 0.5 * self._pml_x)

    def cal_v_par(self, v_par, t_z):
        t_z = t_z / self._pmodel
        return (t_z + (1 / self._dt - 0.5 * self._pml_z) * v_par) / (1 / self._dt + 0.5 * self._pml_z)
    
    def cal_r_ver(self, r_ver, u_x):
        u_x = u_x * self._c11
        return (u_x + (1 / self._dt - 0.5 * self._pml_x) * r_ver) / (1 / self._dt + 0.5 * self._pml_x)

    def cal_r_par(self, r_par, v_z):
        v_z = v_z * self._c13
        return (v_z + (1 / self._dt - 0.5 * self._pml_z) * r_par) / (1 / self._dt + 0.5 * self._pml_z)
    
    def cal_t_ver(self, t_ver, u_x):
        u_x = u_x * self._c13
        return (u_x + (1 / self._dt - 0.5 * self._pml_x) * t_ver) / (1 / self._dt + 0.5 * self._pml_x)

    def cal_t_par(self, t_par, v_z):
        v_z = v_z * self._c11
        return (v_z + (1 / self._dt - 0.5 * self._pml_z) * t_par) / (1 / self._dt + 0.5 * self._pml_z)
    
    def cal_h_ver(self, h_ver, v_x):
        v_x = v_x * self._c44
        return (v_x + (1 / self._dt - 0.5 * self._pml_x) * h_ver) / (1 / self._dt + 0.5 * self._pml_x)

    def cal_h_par(self, h_par, u_z):
        u_z = u_z * self._c44
        return (u_z + (1 / self._dt - 0.5 * self._pml_z) * h_par) / (1 / self._dt + 0.5 * self._pml_z)

    def DrawXRecord(self, amp = 1.0):
        surface_record_no_bc = self._vx[self._pml_len:-self._pml_len,self._pml_len,:]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap = 'gray', vmin = -1, vmax = 1, interpolation='bilinear', aspect='auto')
        plt.title("Vx Record")
        plt.show()

    def DrawZRecord(self, amp = 1.0):
        surface_record_no_bc = self._vz[self._pml_len:-self._pml_len,self._pml_len,:]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap = 'gray', vmin = -1, vmax = 1, interpolation='bilinear', aspect='auto')
        plt.title("Vz Record")
        plt.show()

# In[4]:
class FirstOlder_NPML(FirstOlderForward):
    
    def __init__(self, pmodel, vpmodel, vsmodel, tmax, xmax, zmax, nt, nx, nz, pml_len,\
                 sigma_x_max = 1200, sigma_z_max = 1200, kai_x_max = 2, kai_z_max = 2, alpha_x_max = 150 ,alpha_z_max = 150):
        
        dt = tmax / nt
        super().__init__(pmodel, vpmodel, vsmodel, tmax, xmax, zmax, nt, nx, nz)
        self._pml_len = pml_len
        
        xspacing = np.linspace(0,nx,nx)
        zspacing = np.linspace(0,nz,nz)
        zmesh, xmesh = np.meshgrid(zspacing, xspacing)
        
#         sigma_x_max = 1200
#         sigma_z_max = 1200
#         kai_x_max = 2
#         kai_z_max = 2
#         alpha_x_max = 150
#         alpha_z_max = 150

        sigma_x = np.where(xmesh<pml_len, sigma_x_max * np.power((pml_len-xmesh)/pml_len,2),\
                               np.where(nx-xmesh<pml_len, sigma_x_max * np.power((xmesh-nx+pml_len)/pml_len,2), 0))
        
        sigma_z = np.where(zmesh<pml_len, sigma_z_max * np.power((pml_len-zmesh)/pml_len,2),\
                               np.where(nz-zmesh<pml_len, sigma_z_max * np.power((zmesh-nz+pml_len)/pml_len,2), 0))
        
        kai_x = np.where(xmesh<pml_len, (kai_x_max - 1) * np.power((pml_len-xmesh)/pml_len,2) + 1,\
                               np.where(nx-xmesh<pml_len, (kai_x_max - 1) * np.power((xmesh-nx+pml_len)/pml_len,2) + 1, 1))
        
        kai_z = np.where(zmesh<pml_len, (kai_z_max - 1) * np.power((pml_len-zmesh)/pml_len,2) + 1,\
                               np.where(nz-zmesh<pml_len, (kai_z_max - 1) * np.power((zmesh-nz+pml_len)/pml_len,2) + 1, 1))
        
        alpha_x = np.where(xmesh<pml_len, alpha_x_max * np.power((pml_len-xmesh)/pml_len,2),\
                               np.where(nx-xmesh<pml_len, alpha_x_max * np.power((xmesh-nx+pml_len)/pml_len,2), 0))
        
        alpha_z = np.where(zmesh<pml_len, alpha_z_max * np.power((pml_len-zmesh)/pml_len,2),\
                               np.where(nz-zmesh<pml_len, alpha_z_max * np.power((zmesh-nz+pml_len)/pml_len,2), 0))
        
        self._kai_x = kai_x
        self._kai_z = kai_z
        
#         plt.imshow(sigma_x)
#         plt.colorbar(shrink=0.9)
#         plt.show()    
        
#         plt.imshow(kai_x)
#         plt.colorbar(shrink=0.9)
#         plt.show()        
        
#         plt.imshow(alpha_x)
#         plt.colorbar(shrink=0.9)
#         plt.show()
        
        self._b_x = np.exp(-1 * (alpha_x + sigma_x / kai_x) * dt)
        self._a_x = np.where(sigma_x==0, 0, (1 - self._b_x) * sigma_x / (kai_x * (kai_x * alpha_x + sigma_x)))
        # self._c_x = 1 - 1 / kai_x
        
        self._b_z = np.exp(-1 * (alpha_z + sigma_z / kai_z) * dt)
        self._a_z = np.where(sigma_z==0, 0, (1 - self._b_z) * sigma_z / (kai_z * (kai_z * alpha_z + sigma_z)))
        # self._c_z = 1 - 1 / kai_z
        
    def o2xFM(self, wavelet, wavalet_position, wavalet_direction='z'):
        
        t_array = np.arange(0, self._tmax, self._dt)
        nx = self._nx
        nz = self._nz
        
        # set auxiliary parameters
        u = np.zeros((nx,nz), dtype=float)
        v = np.zeros((nx,nz), dtype=float)
        r = np.zeros((nx,nz), dtype=float)
        t = np.zeros((nx,nz), dtype=float)
        h = np.zeros((nx,nz), dtype=float)
        
        omega_xx = np.zeros((nx,nz), dtype=float)
        omega_xz = np.zeros((nx,nz), dtype=float)
        omega_zx = np.zeros((nx,nz), dtype=float)
        omega_zz = np.zeros((nx,nz), dtype=float)
        phi_xx = np.zeros((nx,nz), dtype=float)
        phi_zz = np.zeros((nx,nz), dtype=float)
        phi_zx = np.zeros((nx,nz), dtype=float)
        phi_xz = np.zeros((nx,nz), dtype=float)

        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute

                u_x = self.o2x_cal_u_x(u)
                u_z = self.o2x_cal_u_z(u)
                v_x = self.o2x_cal_v_x(v)
                v_z = self.o2x_cal_v_z(v)
                
                r = self.cal_r(r, u_x, v_z, phi_xx, phi_zz)
                t = self.cal_t(t, u_x, v_z, phi_xx, phi_zz)
                h = self.cal_h(h, v_x, u_z, phi_zx, phi_xz)
                
                r_x = self.o2x_cal_r_x(r)
                t_z = self.o2x_cal_t_z(t)
                h_x = self.o2x_cal_h_x(h)
                h_z = self.o2x_cal_h_z(h)
                
                # omega_update
                omega_xx = self.cal_omega_xx(r_x, omega_xx)
                omega_xz = self.cal_omega_xz(h_z, omega_xz)
                omega_zx = self.cal_omega_zx(h_x, omega_zx)
                omega_zz = self.cal_omega_zz(t_z, omega_zz)
                
                u = self.cal_u(u, r_x, h_z, omega_xx, omega_xz)
                v = self.cal_v(v, h_x, t_z, omega_zx, omega_zz)
                
                # phi_update
                phi_xx = self.cal_phi_xx(u_x, phi_xx)
                phi_zz = self.cal_phi_zz(v_z, phi_zz)
                phi_zx = self.cal_phi_zx(v_x, phi_zx)
                phi_xz = self.cal_phi_xz(u_z, phi_xz)
                
                if tk < len(wavelet):  # source is active
                    if wavalet_direction=='x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                
            self._vx[:,:,tk] = u
            self._vz[:,:,tk] = v

    def o2anyFM(self, cal_par, wavelet, wavalet_position, wavalet_direction='z'):

        t_array = np.arange(0, self._tmax, self._dt)
        nx = self._nx
        nz = self._nz

        # set auxiliary parameters
        u = np.zeros((nx, nz), dtype=float)
        v = np.zeros((nx, nz), dtype=float)
        r = np.zeros((nx, nz), dtype=float)
        t = np.zeros((nx, nz), dtype=float)
        h = np.zeros((nx, nz), dtype=float)

        omega_xx = np.zeros((nx, nz), dtype=float)
        omega_xz = np.zeros((nx, nz), dtype=float)
        omega_zx = np.zeros((nx, nz), dtype=float)
        omega_zz = np.zeros((nx, nz), dtype=float)
        phi_xx = np.zeros((nx, nz), dtype=float)
        phi_zz = np.zeros((nx, nz), dtype=float)
        phi_zx = np.zeros((nx, nz), dtype=float)
        phi_xz = np.zeros((nx, nz), dtype=float)

        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute

                u_x = self.o2any_cal_u_x(u, cal_par)
                u_z = self.o2any_cal_u_z(u, cal_par)
                v_x = self.o2any_cal_v_x(v, cal_par)
                v_z = self.o2any_cal_v_z(v, cal_par)

                r = self.cal_r(r, u_x, v_z, phi_xx, phi_zz)
                t = self.cal_t(t, u_x, v_z, phi_xx, phi_zz)
                h = self.cal_h(h, v_x, u_z, phi_zx, phi_xz)

                r_x = self.o2any_cal_r_x(r, cal_par)
                t_z = self.o2any_cal_t_z(t, cal_par)
                h_x = self.o2any_cal_h_x(h, cal_par)
                h_z = self.o2any_cal_h_z(h, cal_par)

                # omega_update
                omega_xx = self.cal_omega_xx(r_x, omega_xx)
                omega_xz = self.cal_omega_xz(h_z, omega_xz)
                omega_zx = self.cal_omega_zx(h_x, omega_zx)
                omega_zz = self.cal_omega_zz(t_z, omega_zz)

                u = self.cal_u(u, r_x, h_z, omega_xx, omega_xz)
                v = self.cal_v(v, h_x, t_z, omega_zx, omega_zz)

                # phi_update
                phi_xx = self.cal_phi_xx(u_x, phi_xx)
                phi_zz = self.cal_phi_zz(v_z, phi_zz)
                phi_zx = self.cal_phi_zx(v_x, phi_zx)
                phi_xz = self.cal_phi_xz(u_z, phi_xz)

                if tk < len(wavelet):  # source is active
                    if wavalet_direction == 'x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:, :, tk] = u
            self._vz[:, :, tk] = v
            
    def cal_u(self, u, r_x, h_z, omega_xx, omega_xz):
        right = 1 / self._kai_x * r_x - omega_xx + 1 / self._kai_z * h_z - omega_xz
        return right * self._dt / self._pmodel + u
    
    def cal_v(self, v, h_x, t_z, omega_zx, omega_zz):
        right = 1 / self._kai_x * h_x - omega_zx + 1 / self._kai_z * t_z - omega_zz
        return right * self._dt / self._pmodel + v
            
    def cal_r(self, r, u_x, v_z, phi_xx, phi_zz):
        right = self._c11 * (1 / self._kai_x * u_x - phi_xx) + self._c13 * (1 / self._kai_z * v_z - phi_zz)
        return right * self._dt + r

    def cal_t(self, t, u_x, v_z, phi_xx, phi_zz):
        right = self._c13 * (1 / self._kai_x * u_x - phi_xx) + self._c11 * (1 / self._kai_z * v_z - phi_zz)
        return right * self._dt + t
    
    def cal_h(self, h, v_x, u_z, phi_zx, phi_xz):
        right = self._c44 * (1 / self._kai_x * v_x - phi_zx + 1 / self._kai_z * u_z - phi_xz)
        return right * self._dt + h
    
    def cal_omega_xx(self, r_x, omega_xx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * omega_xx + a_x * r_x
    
    def cal_omega_xz(self, h_z, omega_xz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * omega_xz + a_z * h_z
    
    def cal_omega_zx(self, h_x, omega_zx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * omega_zx + a_x * h_x
    
    def cal_omega_zz(self, t_z, omega_zz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * omega_zz + a_z * t_z
    
    def cal_phi_xx(self, u_x, phi_xx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * phi_xx + a_x * u_x
    
    def cal_phi_zz(self, v_z, phi_zz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * phi_zz + a_z * v_z
        
    def cal_phi_zx(self, v_x, phi_zx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * phi_zx + a_x * v_x
        
    def cal_phi_xz(self, u_z, phi_xz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * phi_xz + a_z * u_z
    
    def DrawXRecord(self, amp = 1.0):
        surface_record_no_bc = self._vx[self._pml_len:-self._pml_len,self._pml_len,:]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap = 'gray', vmin = -1, vmax = 1, interpolation='bilinear', aspect='auto')
        plt.title("Vx Record")
        plt.show()

    def DrawZRecord(self, amp = 1.0):
        surface_record_no_bc = self._vz[self._pml_len:-self._pml_len,self._pml_len,:]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap = 'gray', vmin = -1, vmax = 1, interpolation='bilinear', aspect='auto')
        plt.title("Vz Record")
        plt.show()

# In[5]:
class FirstOlder_NPML_PS(FirstOlder_NPML):

    def __init__(self, pmodel, vpmodel, vsmodel, tmax, xmax, zmax, nt, nx, nz, pml_len, \
                 sigma_x_max=1200, sigma_z_max=1200, kai_x_max=2, kai_z_max=2, alpha_x_max=150, alpha_z_max=150):

        dt = tmax / nt
        super().__init__(pmodel, vpmodel, vsmodel, tmax, xmax, zmax, nt, nx, nz, pml_len, sigma_x_max, sigma_z_max, kai_x_max, kai_z_max, alpha_x_max, alpha_z_max)
        # self._pml_len = pml_len
        self._vxp = np.zeros((nx, nz, nt), dtype=float)
        self._vzp = np.zeros((nx, nz, nt), dtype=float)
        self._vxs = np.zeros((nx, nz, nt), dtype=float)
        self._vzs = np.zeros((nx, nz, nt), dtype=float)

    def o2xFM(self, wavelet, wavalet_position, wavalet_direction='z'):

        t_array = np.arange(0, self._tmax, self._dt)
        nx = self._nx
        nz = self._nz

        # set auxiliary parameters
        u = np.zeros((nx, nz), dtype=float)
        v = np.zeros((nx, nz), dtype=float)
        r = np.zeros((nx, nz), dtype=float)
        t = np.zeros((nx, nz), dtype=float)
        h = np.zeros((nx, nz), dtype=float)

        tau_p = np.zeros((nx, nz), dtype=float)
        vxp = np.zeros((nx, nz), dtype=float)
        vzp = np.zeros((nx, nz), dtype=float)
        vxs = np.zeros((nx, nz), dtype=float)
        vzs = np.zeros((nx, nz), dtype=float)

        omega_xx = np.zeros((nx, nz), dtype=float)
        omega_xz = np.zeros((nx, nz), dtype=float)
        omega_zx = np.zeros((nx, nz), dtype=float)
        omega_zz = np.zeros((nx, nz), dtype=float)
        phi_xx = np.zeros((nx, nz), dtype=float)
        phi_zz = np.zeros((nx, nz), dtype=float)
        phi_zx = np.zeros((nx, nz), dtype=float)
        phi_xz = np.zeros((nx, nz), dtype=float)

        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute

                r = self.cal_r(r, u_x, v_z, phi_xx, phi_zz)
                t = self.cal_t(t, u_x, v_z, phi_xx, phi_zz)
                h = self.cal_h(h, v_x, u_z, phi_zx, phi_xz)

                r_x = self.o2x_cal_r_x(r)
                t_z = self.o2x_cal_t_z(t)
                h_x = self.o2x_cal_h_x(h)
                h_z = self.o2x_cal_h_z(h)

                # omega_update
                omega_xx = self.cal_omega_xx(r_x, omega_xx)
                omega_xz = self.cal_omega_xz(h_z, omega_xz)
                omega_zx = self.cal_omega_zx(h_x, omega_zx)
                omega_zz = self.cal_omega_zz(t_z, omega_zz)

                u = self.cal_u(u, r_x, h_z, omega_xx, omega_xz)
                v = self.cal_v(v, h_x, t_z, omega_zx, omega_zz)

                # phi_update
                phi_xx = self.cal_phi_xx(u_x, phi_xx)
                phi_zz = self.cal_phi_zz(v_z, phi_zz)
                phi_zx = self.cal_phi_zx(v_x, phi_zx)
                phi_xz = self.cal_phi_xz(u_z, phi_xz)

                if tk < len(wavelet):  # source is active
                    if wavalet_direction == 'x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:, :, tk] = u
            self._vz[:, :, tk] = v

            u_x = self.o2x_cal_u_x(u)
            u_z = self.o2x_cal_u_z(u)
            v_x = self.o2x_cal_v_x(v)
            v_z = self.o2x_cal_v_z(v)

            # wave separation
            tau_p += (u_x + v_z) * self._c11 * self._dt
            tau_p_x = self.o2x_cal_tau_p_x(tau_p)
            tau_p_z = self.o2x_cal_tau_p_z(tau_p)
            vxp += tau_p_x * self._dt / self._pmodel
            vzp += tau_p_z * self._dt / self._pmodel
            vxs = u - vxp
            vzs = v - vzp
            self._vxp[:, :, tk] = vxp
            self._vzp[:, :, tk] = vzp
            self._vxs[:, :, tk] = vxs
            self._vzs[:, :, tk] = vzs

    def o2anyFM(self, cal_par, wavelet, wavalet_position, wavalet_direction='z'):

        t_array = np.arange(0, self._tmax, self._dt)
        nx = self._nx
        nz = self._nz

        # set auxiliary parameters
        u = np.zeros((nx, nz), dtype=float)
        v = np.zeros((nx, nz), dtype=float)
        r = np.zeros((nx, nz), dtype=float)
        t = np.zeros((nx, nz), dtype=float)
        h = np.zeros((nx, nz), dtype=float)

        tau_p = np.zeros((nx, nz), dtype=float)
        vxp = np.zeros((nx, nz), dtype=float)
        vzp = np.zeros((nx, nz), dtype=float)
        vxs = np.zeros((nx, nz), dtype=float)
        vzs = np.zeros((nx, nz), dtype=float)

        omega_xx = np.zeros((nx, nz), dtype=float)
        omega_xz = np.zeros((nx, nz), dtype=float)
        omega_zx = np.zeros((nx, nz), dtype=float)
        omega_zz = np.zeros((nx, nz), dtype=float)
        phi_xx = np.zeros((nx, nz), dtype=float)
        phi_zz = np.zeros((nx, nz), dtype=float)
        phi_zx = np.zeros((nx, nz), dtype=float)
        phi_xz = np.zeros((nx, nz), dtype=float)

        for tk, tt in enumerate(t_array):
            if tk >= 1:  # the first step needs not to compute

                r = self.cal_r(r, u_x, v_z, phi_xx, phi_zz)
                t = self.cal_t(t, u_x, v_z, phi_xx, phi_zz)
                h = self.cal_h(h, v_x, u_z, phi_zx, phi_xz)

                r_x = self.o2any_cal_r_x(r, cal_par)
                t_z = self.o2any_cal_t_z(t, cal_par)
                h_x = self.o2any_cal_h_x(h, cal_par)
                h_z = self.o2any_cal_h_z(h, cal_par)

                # omega_update
                omega_xx = self.cal_omega_xx(r_x, omega_xx)
                omega_xz = self.cal_omega_xz(h_z, omega_xz)
                omega_zx = self.cal_omega_zx(h_x, omega_zx)
                omega_zz = self.cal_omega_zz(t_z, omega_zz)

                u = self.cal_u(u, r_x, h_z, omega_xx, omega_xz)
                v = self.cal_v(v, h_x, t_z, omega_zx, omega_zz)

                # phi_update
                phi_xx = self.cal_phi_xx(u_x, phi_xx)
                phi_zz = self.cal_phi_zz(v_z, phi_zz)
                phi_zx = self.cal_phi_zx(v_x, phi_zx)
                phi_xz = self.cal_phi_xz(u_z, phi_xz)

                if tk < len(wavelet):  # source is active
                    if wavalet_direction == 'x':
                        u[wavalet_position[1], wavalet_position[0]] += wavelet[tk]
                    else:
                        v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

            self._vx[:, :, tk] = u
            self._vz[:, :, tk] = v

            u_x = self.o2any_cal_u_x(u, cal_par)
            u_z = self.o2any_cal_u_z(u, cal_par)
            v_x = self.o2any_cal_v_x(v, cal_par)
            v_z = self.o2any_cal_v_z(v, cal_par)

            # wave separation
            tau_p += (u_x + v_z) * self._c11 * self._dt
            tau_p_x = self.o2any_cal_tau_p_x(tau_p, cal_par)
            tau_p_z = self.o2any_cal_tau_p_z(tau_p, cal_par)
            vxp += tau_p_x * self._dt / self._pmodel
            vzp += tau_p_z * self._dt / self._pmodel
            vxs = u - vxp
            vzs = v - vzp
            self._vxp[:, :, tk] = vxp
            self._vzp[:, :, tk] = vzp
            self._vxs[:, :, tk] = vxs
            self._vzs[:, :, tk] = vzs

    def o2x_cal_tau_p_x(self, tau_p):
        kernal = np.zeros((11,11), dtype=float)
        kernal[:,5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5]
        return convolve(tau_p, kernal, mode='same') / self._dx

    def o2x_cal_tau_p_z(self, tau_p):
        kernal = np.zeros((11,11),dtype=float)
        kernal[:,5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4, -self._c5, 0]
        kernal = kernal.T
        return convolve(tau_p, kernal, mode='same') / self._dz

    def o2any_cal_tau_p_x(self, tau_p, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[1:array_mid+1, array_mid] = cal_par[::-1]
        kernal[array_mid+1:, array_mid] = cal_par * -1
        # kernal[:, 5] = [0, self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5]
        return convolve(tau_p, kernal, mode='same') / self._dx

    def o2any_cal_tau_p_z(self, tau_p, cal_par):
        array_mid = cal_par.size
        array_size = cal_par.size * 2 + 1
        kernal = np.zeros((array_size, array_size), dtype=float)
        kernal[0:array_mid, array_mid] = cal_par[::-1]
        kernal[array_mid:-1, array_mid] = cal_par * -1
        # kernal = np.zeros((11, 11), dtype=float)
        # kernal[:, 5] = [self._c5, self._c4, self._c3, self._c2, self._c1, -self._c1, -self._c2, -self._c3, -self._c4,
        #                 -self._c5, 0]
        kernal = kernal.T
        return convolve(tau_p, kernal, mode='same') / self._dz

    def cal_u(self, u, r_x, h_z, omega_xx, omega_xz):
        right = 1 / self._kai_x * r_x - omega_xx + 1 / self._kai_z * h_z - omega_xz
        return right * self._dt / self._pmodel + u

    def cal_v(self, v, h_x, t_z, omega_zx, omega_zz):
        right = 1 / self._kai_x * h_x - omega_zx + 1 / self._kai_z * t_z - omega_zz
        return right * self._dt / self._pmodel + v

    def cal_r(self, r, u_x, v_z, phi_xx, phi_zz):
        right = self._c11 * (1 / self._kai_x * u_x - phi_xx) + self._c13 * (1 / self._kai_z * v_z - phi_zz)
        return right * self._dt + r

    def cal_t(self, t, u_x, v_z, phi_xx, phi_zz):
        right = self._c13 * (1 / self._kai_x * u_x - phi_xx) + self._c11 * (1 / self._kai_z * v_z - phi_zz)
        return right * self._dt + t

    def cal_h(self, h, v_x, u_z, phi_zx, phi_xz):
        right = self._c44 * (1 / self._kai_x * v_x - phi_zx + 1 / self._kai_z * u_z - phi_xz)
        return right * self._dt + h

    def cal_omega_xx(self, r_x, omega_xx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * omega_xx + a_x * r_x

    def cal_omega_xz(self, h_z, omega_xz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * omega_xz + a_z * h_z

    def cal_omega_zx(self, h_x, omega_zx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * omega_zx + a_x * h_x

    def cal_omega_zz(self, t_z, omega_zz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * omega_zz + a_z * t_z

    def cal_phi_xx(self, u_x, phi_xx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * phi_xx + a_x * u_x

    def cal_phi_zz(self, v_z, phi_zz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * phi_zz + a_z * v_z

    def cal_phi_zx(self, v_x, phi_zx):
        b_x = self._b_x
        a_x = self._a_x
        return b_x * phi_zx + a_x * v_x

    def cal_phi_xz(self, u_z, phi_xz):
        b_z = self._b_z
        a_z = self._a_z
        return b_z * phi_xz + a_z * u_z

    def DrawXPRecord(self, amp=1.0):
        surface_record_no_bc = self._vxp[self._pml_len:-self._pml_len, self._pml_len, :]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap='gray', vmin=-1, vmax=1,
                   interpolation='bilinear', aspect='auto')
        plt.title("P Wave Record in X direction")
        plt.show()

    def DrawZPRecord(self, amp=1.0):
        surface_record_no_bc = self._vzp[self._pml_len:-self._pml_len, self._pml_len, :]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap='gray', vmin=-1, vmax=1,
                   interpolation='bilinear', aspect='auto')
        plt.title("P Wave Record in Z direction")
        plt.show()

    def DrawXSRecord(self, amp=1.0):
        surface_record_no_bc = self._vxs[self._pml_len:-self._pml_len, self._pml_len, :]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap='gray', vmin=-1, vmax=1,
                   interpolation='bilinear', aspect='auto')
        plt.title("S Wave Record in X direction")
        plt.show()

    def DrawZSRecord(self, amp=1.0):
        surface_record_no_bc = self._vzs[self._pml_len:-self._pml_len, self._pml_len, :]
        plt.imshow(self.RangeInOne(surface_record_no_bc).T * amp, cmap='gray', vmin=-1, vmax=1,
                   interpolation='bilinear', aspect='auto')
        plt.title("S Wave Record in Z direction")
        plt.show()

    def DrawXPWaveField(self, iterations_to_show):
        fgr, axs = plt.subplots(1,len(iterations_to_show), figsize = (18,10))
        for j, ax in enumerate(axs):
            ax.imshow(self.RangeInOne(self._vxp[:, :, iterations_to_show[j]].T), cmap = plt.cm.coolwarm, vmin = -1, vmax = 1, interpolation='bilinear')
            ax.annotate("t = {0} ms".format(iterations_to_show[j]), xy=(0.05, 0.05), xycoords="axes fraction")
        plt.show()

    def DrawXSWaveField(self, iterations_to_show):
        fgr, axs = plt.subplots(1,len(iterations_to_show), figsize = (18,10))
        for j, ax in enumerate(axs):
            ax.imshow(self.RangeInOne(self._vxs[:, :, iterations_to_show[j]].T), cmap = plt.cm.coolwarm, vmin = -1, vmax = 1, interpolation='bilinear')
            ax.annotate("t = {0} ms".format(iterations_to_show[j]), xy=(0.05, 0.05), xycoords="axes fraction")
        plt.show()

    def DrawZPWaveField(self, iterations_to_show):
        fgr, axs = plt.subplots(1, len(iterations_to_show), figsize=(18, 10))
        for j, ax in enumerate(axs):
            ax.imshow(self.RangeInOne(self._vzp[:, :, iterations_to_show[j]].T), cmap=plt.cm.coolwarm, vmin=-1, vmax=1,
                      interpolation='bilinear')
            ax.annotate("t = {0} ms".format(iterations_to_show[j]), xy=(0.05, 0.05), xycoords="axes fraction")
        plt.show()

    def DrawZSWaveField(self, iterations_to_show):
        fgr, axs = plt.subplots(1, len(iterations_to_show), figsize=(18, 10))
        for j, ax in enumerate(axs):
            ax.imshow(self.RangeInOne(self._vzs[:, :, iterations_to_show[j]].T), cmap=plt.cm.coolwarm, vmin=-1, vmax=1,
                      interpolation='bilinear')
            ax.annotate("t = {0} ms".format(iterations_to_show[j]), xy=(0.05, 0.05), xycoords="axes fraction")
        plt.show()