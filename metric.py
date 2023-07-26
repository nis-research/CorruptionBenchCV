

import numpy as np
import scipy.special as ss
import pandas as pd

def mCE(CEs):
    # CEs: a list of corruption error
    return sum(CEs)/len(CEs)

def CE(err_test, err_baseline):
    # err_test: a list of classification error of model tested on a corruption with 5 levels of severity
    return sum(err_test) / sum(err_baseline) 

def get_acc(csv_test):
    resutls_test = pd.read_csv(csv_test)

    Corruptions_list = resutls_test['Corruption']
    Corruptions_list = list(Corruptions_list)

    acc_test = []
    for corruption in Corruptions_list:
        corruption_i = Corruptions_list.index(corruption)
        for severity in range(5):
            acc_test.append(resutls_test.loc[corruption_i,'Acc_s'+str(severity+1)])
           

    return sum(acc_test)/len(acc_test)

   

def RelativeRobustness(acc_test, acc_baseline):
    return acc_test-acc_baseline

def get_mCE(csv_test,csv_base):
    resutls_test = pd.read_csv(csv_test)
    resutls_baseline = pd.read_csv(csv_base)

    Corruptions_list = resutls_test['Corruption']
    Corruptions_list = list(Corruptions_list)

    x = [1,2,3,4,5]
    mCEs = []
    for corruption in Corruptions_list:
        err_test = []
        err_baseline = []
        corruption_i = Corruptions_list.index(corruption)
        for severity in range(5):
            err_test.append(100-resutls_test.loc[corruption_i,'Acc_s'+str(severity+1)])
            err_baseline.append(100-resutls_baseline.loc[corruption_i,'Acc_s'+str(severity+1)])

        CE_i = CE(err_test,err_baseline)
        mCEs.append(CE_i)
    result_mCE = mCE(mCEs)
    return result_mCE*100.0

def rCE(err_test,err_baseline,err_test_clean,err_baseline_clean):
    err_test = np.asarray(err_test)
    err_baseline = np.asarray(err_baseline)
    err_test_clean = np.asarray(err_test_clean)
    err_baseline_clean = np.asarray(err_baseline_clean)
    # print((err_test-err_test_clean))
    # print((err_baseline-err_baseline_clean))
    # print(sum((err_test-err_test_clean)))
    # print(sum((err_baseline-err_baseline_clean)))
    return sum((err_test-err_test_clean))/sum((err_baseline-err_baseline_clean))

def mrCE(rCEs):
    return sum(rCEs)/len(rCEs)

def get_rCE(csv_test,csv_base, err_test, err_clean):
    resutls_test = pd.read_csv(csv_test)
    resutls_baseline = pd.read_csv(csv_base)

    Corruptions_list = resutls_test['Corruption']
    Corruptions_list = list(Corruptions_list)

    x = [1,2,3,4,5]
    rCEs = []
    err_test_clean = [err_test,err_test,err_test,err_test,err_test] # change for a different test model
    err_baseline_clean = [err_clean,err_clean,err_clean,err_clean,err_clean]
    for corruption in Corruptions_list:
        err_test = []
        err_baseline = []
        corruption_i = Corruptions_list.index(corruption)
        for severity in range(5):
            err_test.append(100-resutls_test.loc[corruption_i,'Acc_s'+str(severity+1)])
            err_baseline.append(100-resutls_baseline.loc[corruption_i,'Acc_s'+str(severity+1)])

        RCE = rCE(err_test,err_baseline,err_test_clean,err_baseline_clean)
        rCEs.append(RCE)
    result_rCE = mrCE(rCEs)
    return result_rCE*100.0


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
 
def get_ece(csv_test,csv_base):
    resutls_test = pd.read_csv(csv_test)
    resutls_baseline = pd.read_csv(csv_base)

    Corruptions_list = resutls_test['Corruption']
    Corruptions_list = list(Corruptions_list)
 
    eces = []
    for corruption in Corruptions_list:
        ece_test = []
        ece_baseline = []
        corruption_i = Corruptions_list.index(corruption)
        for severity in range(5):
            ece_test.append(resutls_test.loc[corruption_i,'ECE_s'+str(severity+1)])
            ece_baseline.append(resutls_baseline.loc[corruption_i,'ECE_s'+str(severity+1)])

        ece_i = sum(ece_test) / sum(ece_baseline) 
        eces.append(ece_i)
    result_ECE = sum(eces)/len(eces)
    return result_ECE

def get_mfp_mt5d(csv_test,csv_base):
    resutls_test = pd.read_csv(csv_test)
    resutls_baseline = pd.read_csv(csv_base)

    Corruptions_list = resutls_test['Corruption']
    Corruptions_list = list(Corruptions_list)
    fps = []
    t5ds = []
    for corruption in Corruptions_list:
   
        corruption_i = Corruptions_list.index(corruption)
        
        fps.append(resutls_test.loc[corruption_i,'FP']/resutls_baseline.loc[corruption_i,'FP'])
        t5ds.append(resutls_test.loc[corruption_i,'T5D']/resutls_baseline.loc[corruption_i,'T5D'])
  
    result_mFP = sum(fps)/len(fps)*100.0
    result_t5ds = sum(t5ds)/len(t5ds)*100.0

    return result_mFP,result_t5ds
