#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from scipy.signal import convolve


# In[19]:


# # 2nd in time
# class Operator():
#     # data
    
#     # function
#     def __init__(self, model, source, receiver, ft, order, coefficient):
#         self._model = model
#         self._source = source
#         self._receiver = receiver
#         self._ft = ft
#         return True
        
#     def execute(self,):
        
#         array_size = self._model._array_size
#         t_array = ft._t_array
        
#         u = np.zeros(array_size, dtype=float)
#         v = np.zeros(array_size, dtype=float)
#         r = np.zeros(array_size, dtype=float)
#         t = np.zeros(array_size, dtype=float)
#         h = np.zeros(array_size, dtype=float)
        
#         for tk, tt in enumerate(t_array):
#             if tk >= 1:  # the first step needs not to compute
                
#                 u_x = self.o24_cal_u_x(u)
#                 u_z = self.o24_cal_u_z(u)
#                 v_x = self.o24_cal_v_x(v)
#                 v_z = self.o24_cal_v_z(v)
                
#                 r = self.o24ComputeTauxx(r, u_x ,v_z)
#                 t = self.o24ComputeTauzz(t, u_x ,v_z)
#                 h = self.o24ComputeTauxz(h ,v_x, u_z)

#                 r_x = self.o24_cal_r_x(r)
#                 t_z = self.o24_cal_t_z(t)
#                 h_x = self.o24_cal_h_x(h)
#                 h_z = self.o24_cal_h_z(h)

#                 u = self.o24ComputeVx(u, r_x, h_z)
#                 v = self.o24ComputeVz(v, t_z, h_x)
                

                
#                 if tk < len(wavelet):  # source is active
#                     if wavalet_direction=='x':
#                         u[wavalet_position[1], wavalet_position[0]] += wavelet[tk] 
#                     else:
#                         v[wavalet_position[1], wavalet_position[0]] += wavelet[tk]

#             self._vx[:,:,tk] = u
#             self._vz[:,:,tk] = v
            
#             if (np.max(u)>20):
#                 print("divergent! Please reset gird spacing or/and time step length.")
#                 return
        
#         return True


# In[26]:


class Operator():
    # data
    
    # function
    # older =  (t, s) t:time older, s:space order
    # coefficien = [c1, c2, c3, ...] used for computing derivative
    # space_step = (dx, dz)
    def __init__(self, older, coefficient, space_step):
        self._to = older[0]
        self._so = older[1]
        self._dx = space_step[0]
        self._dz = space_step[1]
        self._coefficient = coefficient
        
    def compute_u_x(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=np.float32)
        kernal[0:mid_position,mid_position] = coefficient[::-1] 
        kernal[mid_position:-1,mid_position] = coefficient * -1
        return convolve(data, kernal, mode='same') / self._dx
    
    def compute_u_z(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[mid_position,0:mid_position] = coefficient[::-1] 
        kernal[mid_position,mid_position:-1] = coefficient * -1
        return convolve(data, kernal, mode='same') / self._dz
    
    def compute_v_x(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[1:mid_position+1,mid_position] = coefficient[::-1] 
        kernal[mid_position+1:,mid_position] = coefficient * -1
        return convolve(data, kernal, mode='same') / self._dx
  
    def compute_v_z(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[mid_position,1:mid_position+1] = coefficient[::-1] 
        kernal[mid_position,mid_position+1:] = coefficient * -1
        return convolve(data, kernal, mode='same') / self._dz
    
    def compute_r_x(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[1:mid_position+1,mid_position] = coefficient[::-1] 
        kernal[mid_position+1:,mid_position] = coefficient * -1
        return convolve(data, kernal, mode='same') / self._dx
    
    def compute_h_z(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[1:mid_position+1,mid_position] = coefficient[::-1] 
        kernal[mid_position+1:,mid_position] = coefficient * -1
        kernal = kernal.T
        return convolve(data, kernal, mode='same') / self._dz
    
    def compute_h_x(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[0:mid_position,mid_position] = coefficient[::-1] 
        kernal[mid_position:-1,mid_position] = coefficient * -1
        return convolve(data, kernal, mode='same') / self._dx
    
    def compute_t_z(self, data):
        coefficient = np.array(self._coefficient)
        mid_position = coefficient.size
        kernal_size = coefficient.size * 2 + 1
        kernal = np.zeros((kernal_size, kernal_size), dtype=float)
        kernal[0:mid_position,mid_position] = coefficient[::-1] 
        kernal[mid_position:-1,mid_position] = coefficient * -1
        kernal = kernal.T        
        return convolve(data, kernal, mode='same') / self._dz
