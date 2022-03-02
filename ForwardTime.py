#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import numpy as np


# In[3]:


class ForwardTime():
    # data 
    
    # function
    def __init__(self, dt, nt, t_len):
        self._dt = dt
        self._nt = nt
        self._t_len = t_len
        self._t_array = np.linspace(0,t_len, nt)