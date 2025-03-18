# nohup python -u evaluation.py --dataset "amazon" > evaluation.log 2>&1 &
# nohup python -u evaluation.py --dataset "bbcsport" > evaluation.log 2>&1 &

import numpy as np
import pandas as pd
import argparse
from sklearn.neighbors import KNeighborsClassifier  


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="amazon", help="Which dataset to use")
parser.add_argument("--ntry", type=int, default=10, help="number of try")
args = parser.parse_args()




y = np.load('./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_poincare_y.npy')

sfw_dmatrix = np.load('./save_results/'+args.dataset+'/sfw.npy')
M = sfw_dmatrix.shape[0]
res = np.zeros(args.ntry)

for i in range(args.ntry):
    M1 = int(M*0.8)
    M2 = M-M1
    np.random.seed(i*77)
    itrain = np.random.choice(M,M1,replace=False)
    itest = list(set(range(M)) - set(itrain))
    sfw_dmatrix1 = sfw_dmatrix[:,itrain]
    sfw_dmatrix1 = sfw_dmatrix1[itrain,:]

    sfw_dmatrix2 = sfw_dmatrix[itest,:]
    sfw_dmatrix2 = sfw_dmatrix2[:,itrain]
    train_label = y[itrain]
    test_label = y[itest]

    estimator = KNeighborsClassifier(metric='precomputed',n_neighbors=20)
    estimator.fit(sfw_dmatrix1,train_label)
    res[i] = np.sum(estimator.predict(sfw_dmatrix2)==test_label)/M2

print("sfw : ", np.mean(res), '(', np.std(res),')')





hhsw_dmatrix = np.load('./save_results/'+args.dataset+'/hhsw.npy')
res = np.zeros(args.ntry)

for i in range(args.ntry):
    M1 = int(M*0.8)
    M2 = M-M1
    np.random.seed(77*i)
    itrain = np.random.choice(M,M1,replace=False)
    itest = list(set(range(M)) - set(itrain))
    hhsw_dmatrix1 = hhsw_dmatrix[:,itrain]
    hhsw_dmatrix1 = hhsw_dmatrix1[itrain,:]
    hhsw_dmatrix2 = hhsw_dmatrix[itest,:]
    hhsw_dmatrix2 = hhsw_dmatrix2[:,itrain]
    train_label = y[itrain]
    test_label = y[itest]

    estimator = KNeighborsClassifier(metric='precomputed',n_neighbors=20)
    estimator.fit(hhsw_dmatrix1,train_label)
    res[i] = np.sum(estimator.predict(hhsw_dmatrix2)==test_label)/M2


print("HHSW : ", np.mean(res), '(', np.std(res),')')





y = np.load('./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_euclidean_y.npy')

hcp_dmatrix = np.load('./save_results/'+args.dataset+'/hcp.npy')
res = np.zeros(args.ntry)

for i in range(args.ntry):
    M1 = int(M*0.8)
    M2 = M-M1
    np.random.seed(i*77)
    itrain = np.random.choice(M,M1,replace=False)
    itest = list(set(range(M)) - set(itrain))
    hcp_dmatrix1 = hcp_dmatrix[:,itrain]
    hcp_dmatrix1 = hcp_dmatrix1[itrain,:]

    hcp_dmatrix2 = hcp_dmatrix[itest,:]
    hcp_dmatrix2 = hcp_dmatrix2[:,itrain]
    train_label = y[itrain]
    test_label = y[itest]

    estimator = KNeighborsClassifier(metric='precomputed',n_neighbors=20)
    estimator.fit(hcp_dmatrix1,train_label)
    res[i] = np.sum(estimator.predict(hcp_dmatrix2)==test_label)/M2

print("HCP : ", np.mean(res), '(', np.std(res),')')
