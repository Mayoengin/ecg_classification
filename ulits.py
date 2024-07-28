#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,classification_report
import os
import deepdish.io as ddio 


# In[2]:


def mkdir_recursive(path):
    if path == "":
        return
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        print("creating directory" + path)
        os.mkdir(path)


# In[3]:


def loaddata(input_size, feature):
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    trainlabelData = ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(trainlabelData[feature])
    att = np.concatenate((X, y), axis=1)
    np.random.shuffle(att)
    X, y = att[:, :input_size], att[:, input_size:]
    
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    valData = ddio.load('dataset/test.hdf5')
    vallabelData = ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    
    print("Shape of Xval:", Xval.shape)
    print("Shape of yval:", yval.shape)

    return X, y, Xval, yval


# In[4]:


def loaddata_nosplit(input_size, feature):
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata.hdf5')
    trainlabelData = ddio.load('dataset/labeldata.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(trainlabelData[feature])
    att = np.concatenate((X, y), axis=1)  # Corrected typo
    np.random.shuffle(att)
    X, y = att[:, :input_size], att[:, input_size:]
    return X, y


# In[5]:


def print_results(config, model, Xval, yval, classes):
    # Ensure yval is in the correct shape for evaluation
    yval_reshaped = np.argmax(yval, axis=-1)  # Assuming yval is one-hot encoded
    
    # Predict the validation set
    y_pred = model.predict(Xval)
    y_pred_classes = np.argmax(y_pred, axis=-1)  # Convert predictions to class indices
    
    # Identify unique predicted classes
    unique_pred_classes = np.unique(y_pred_classes)
    
    # Adjust the classes list if it doesn't match the unique predicted classes
    if len(unique_pred_classes) != len(classes):
        print(f"Warning: Number of predicted classes ({len(unique_pred_classes)}) does not match target names ({len(classes)}).")
        print(f"Predicted classes: {unique_pred_classes}")
        
        # Filter out the unused class names
        classes = [classes[i] for i in unique_pred_classes]
    
    # Calculate and print classification report
    report = classification_report(yval_reshaped, y_pred_classes, target_names=classes, output_dict=True, labels=unique_pred_classes)
    print("Class\tPrecision\tRecall\tF1 Score\tSupport")
    
    # Print out the metrics for each class including support (the number of true instances for each label)
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{label}\t{metrics['precision']:.2f}\t\t{metrics['recall']:.2f}\t{metrics['f1-score']:.2f}\t\t{metrics['support']}")
    
    # Print macro average (excluding the last line for 'accuracy' as it is overall accuracy)
    macro_avg = report['macro avg']
    print(f"Macro Avg\t{macro_avg['precision']:.2f}\t\t{macro_avg['recall']:.2f}\t{macro_avg['f1-score']:.2f}\t\t-")
    
    # If you want to include accuracy per class, you need to calculate it manually using the confusion matrix
    cm = confusion_matrix(yval_reshaped, y_pred_classes)
    accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(classes):
        print(f"Accuracy for class {label}: {accuracies[i]:.2f}")

# In[ ]:




