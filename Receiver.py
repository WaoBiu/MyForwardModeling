#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from ForwardTime import *
import matplotlib.pyplot as plt


# In[7]:


class Receiver():
    # data
    
    # functions
    def __init__(self, total_time, position):
        nt = total_time._nt
        record_array = np.zeros(nt, dtype=float)
        self._position = position
        self._rax = record_array
        self._raz = record_array
        self._ta = total_time
    
    # to be continued
    def draw_record(self):
        return 


# In[ ]:




