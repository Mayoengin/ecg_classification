#!/usr/bin/env python
# coding: utf-8

# In[5]:


from  __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
import random
from utils import*
from config import get_config


# In[6]:


config = get_config()


# In[8]:


num='100'
features =['MLII','V1','V2','V4','V5']
datasetname = 'test/train.hdf5'
labelsname = 'test/trainlabel.hdf5'
classes = ['N', 'V', '/','R','A','L','S']
Nclass = len(classes)
datadict,datalabel = dict(), dict()

for feature in features:
    datadict[feature] = list()
    datalabel[feature] = list()
    
input_size = config.input_size
    


# In[9]:


from wfdb import rdrecord , rdann
record = rdrecord('dataset/'+ num,smooth_frames=True)

from sklearn import preprocessing
signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0]))
signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1]))


# In[10]:


from scipy.signal import find_peaks
peaks,_ = find_peaks(signals0,distance=150)
feature0, feature1 = record.sig_name[0],record.sig_name[1]

for peak in peaks[1:-1]:
    start,end = peak-input_size//2,peak+input_size//2
    ann = rdann('dataset/' + num, extension='atr', sampfrom=start, sampto=end, return_label_elements=['symbol'])

    def to_dict(chosenSym):
        y = [0] * Nclass
        y[classes.index(chosenSym)] = 1
        datalabel[feature0].append(y)
        datalabel[feature1].append(y)
        datadict[feature0].append(signals0[start:end])
        datadict[feature1].append(signals1[start:end])

    annSymbol = ann.symbol

    if len(annSymbol) == 1 and (annSymbol[0] in classes):
        to_dict(annSymbol[0])    


# In[11]:


for feature in ["MLII", "V1"]:
    datadict[feature] = np.array(datadict[feature])
    datalabel[feature] = np.array(datalabel[feature])


# In[ ]:





# In[ ]:




