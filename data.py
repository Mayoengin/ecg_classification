    #!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
import random
from config import get_config
from wfdb import rdrecord, rdann
from scipy.signal import find_peaks


# In[2]:


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess(split):
    nums = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119',
            '121', '122', '123', '124', '200', '201', '202', '203', '205',
            '207', '208', '209', '210', '212', '213', '214', '215', '217', '219',
            '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
    
    features = ['MLII', 'V1', 'V2', 'V3', 'V4', 'V5']
    
    if split:
        testset = ['101', '105', '114', '118', '124', '201', '210', '217']
        trainset = [x for x in nums if x not in testset]
        
        dataSaver(trainset, 'dataset/trainseee.hdf5', 'dataset/trainlabelsee.hdf5', features)
        dataSaver(testset, 'dataset/testseee.hdf5', 'dataset/testlabelsee.hdf5', features)
    else:
        num = 'sample_num'  
        dataSaver(num, 'dataset/targetdatasee.hdf5', 'dataset/labeldatasee.hdf5', features)

def dataSaver(dataSet, datasetname, labelsname, features):
    print("Inside dataSaver")

    classes = ['N', 'V', '/','R','A','L','S']
    Nclass = len(classes)
    datadict, datalabel = dict(), dict()

    for feature in features:
        datadict[feature] = list()
        datalabel[feature] = list()

    def dataprocess():
            input_size = config.input_size
            for num in tqdm(dataSet):
                from wfdb import rdrecord, rdann
                record = rdrecord('dataset/' + num, smooth_frames=True)
                from sklearn import preprocessing
                signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 0])).tolist()
                signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 1])).tolist()
                peaks, _ = find_peaks(signals0, distance=150)
                feature0, feature1 = record.sig_name[0], record.sig_name[1]

                for peak in peaks[1:-1]:
                    start, end = peak - input_size // 2, peak + input_size // 2
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

    dataprocess()

    for feature in ["MLII", "V1"]:
        datadict[feature] = np.array(datadict[feature])
        datalabel[feature] = np.array(datalabel[feature])

    import deepdish as dd

    
    dd.io.save(datasetname, datadict)
    dd.io.save(labelsname, datalabel)
    print("Shape of data saved to", datasetname, ":", datadict["MLII"].shape)
    print("Shape of labels saved to", labelsname, ":", datalabel["MLII"].shape)

def main(config):
    preprocess(config.split)

if __name__ == "__main__":
    config = get_config()
    main(config)


# In[ ]:





# In[ ]:




