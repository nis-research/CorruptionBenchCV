import argparse
import pandas as pd
import matplotlib.pyplot as plt
from metric import rCE, mrCE

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

def main(args):
    # for baseline model: resnet18
    results = get_rCE(args.results_test,args.resutls_baseline,args.err,31.23)
    print(results)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--results_test', type=str, default= 'IN_C/summaryImageNet_C_resnet50.csv',
                    help='resutls of test models')
    parser.add_argument('--resutls_baseline', type=str,default='IN_C/summaryImageNet_C_resnet18.csv',
                    help='resutls of the baseline')
    parser.add_argument('--err', type=float,
                    help='error rate of the test model on clean dataset')



    args = parser.parse_args()

    main(args)