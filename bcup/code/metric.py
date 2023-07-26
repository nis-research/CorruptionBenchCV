

import numpy as np
import scipy.special as ss

def mCE(CEs):
    # CEs: a list of corruption error
    return sum(CEs)/len(CEs)

def CE(err_test, err_baseline):
    # err_test: a list of classification error of model tested on a corruption with 5 levels of severity
    return sum(err_test) / sum(err_baseline) * 100

def RelativeRobustness(acc_test, acc_baseline):
    return acc_test-acc_baseline



def ece_score(py, y_test, n_bins=10):
    
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    # print(py_index)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(ss.softmax(py[i])[py_index[i]])
    py_value = np.array(py_value)
    # print(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
        
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece, Bm

def ECE(ece,Bm):
    return ece/sum(Bm)