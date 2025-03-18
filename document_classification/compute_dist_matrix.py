# nohup python -u compute_dist_matrix.py --embedding "poincare" --dataset "amazon" --method "SFW" > compute_dist_matrix.log 2>&1 &
# nohup python -u compute_dist_matrix.py --embedding "euclidean" --dataset "amazon" --method "HCP" > compute_dist_matrix.log 2>&1 &


import numpy as np
import pandas as pd
import ot
import sys
sys.path.append("../lib")
from utility import * 
from utils_hyperbolic import *
from hhsw import horo_hyper_sliced_wasserstein_poincare
from hsw import hyper_sliced_wasserstein
import torch
import pickle
import argparse
import time
import base


parser = argparse.ArgumentParser()
parser.add_argument("--embedding", type=str, default="poincare", help="Which word2vec to use")
parser.add_argument("--dataset", type=str, default="amazon", help="Which dataset to use")
parser.add_argument("--method", type=str, default="SFW", help="Which method to use")
args = parser.parse_args()


print('Dataset is ', args.dataset)
print('Method is ', args.method)

path = './preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_' + args.embedding + '_w.pkl'    
f = open(path,'rb')
BOW_X = pickle.load(f)

path='./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_' + args.embedding + '_X.pkl'    
f = open(path,'rb')
X = pickle.load(f)

y = np.load('./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_' + args.embedding + '_y.npy')

device = "cuda"
M = X.shape[0]




if args.method=="HHSW":
    
    start_time = time.time()
    hhsw_dist_matrix = np.zeros((M,M))

    for i in range(M):
        for j in range(i+1,M):
            Xs = torch.tensor(X[i].T, dtype=torch.float, device=device)
            Xt = torch.tensor(X[j].T, dtype=torch.float, device=device)

            wa = BOW_X[i]/np.sum(BOW_X[i])
            wa = torch.tensor(wa, dtype=torch.float, device=device)
            wb = BOW_X[j]/np.sum(BOW_X[j])
            wb = torch.tensor(wb, dtype=torch.float, device=device)

            hhsw_dist_matrix[i,j] = horo_hyper_sliced_wasserstein_poincare(Xs, Xt, 500, device, wa, wb, p=2)
        
        if i%10==0:
            print(i/M)

    hhsw_dmatrix = hhsw_dist_matrix+hhsw_dist_matrix.T
    np.save('./save_results/'+args.dataset+'/hhsw.npy', hhsw_dmatrix)
    running_time = time.time() - start_time
    print('Time(HHSW) is ', running_time)




elif args.method=="HCP":
    
    start_time = time.time()
    hcp_dist_matrix = np.zeros((M,M))
    X_sorted = []
    BOW_X_sorted = []

    for i in range(M):
        index = base.hilbert_order(X[i].T)
        X_sorted.append(torch.tensor(X[i].T, dtype=torch.float, device=device)[index,:])
        w = BOW_X[i][index]
        w = w/np.sum(w)
        BOW_X_sorted.append(w)

    for i in range(M):
        for j in range(i+1,M):
            GI = general_plan(BOW_X_sorted[i], BOW_X_sorted[j])
            hcp_dist_matrix[i,j] =  torch.dot( torch.sum(torch.pow( X_sorted[i][GI[:,1].astype(int)]-X_sorted[j][GI[:,2].astype(int)], 2), 1), torch.tensor(GI[:,0]).type(torch.FloatTensor).to(device=device))
       
        if i%10==0:
            print(i/M)

    hcp_dmatrix = hcp_dist_matrix+hcp_dist_matrix.T
    np.save('./save_results/'+args.dataset+'/hcp.npy',hcp_dmatrix)
    running_time = time.time() - start_time
    print('Time(HCP) is ', running_time)




elif args.method=="SFW":

    start_time = time.time()
    sfw_dist_matrix = np.zeros((M,M))
    X_sorted = []
    BOW_X_sorted = []

    for i in range(M):
        index = base.hilbert_order(X[i].T)
        index = base.morton_order(X[i].T)
        X_sorted.append(torch.tensor(X[i].T, dtype=torch.float, device=device)[index,:])
        w = BOW_X[i][index]
        w = w/np.sum(w)
        BOW_X_sorted.append(w)

    for i in range(M):
        for j in range(i+1,M):
            GI = general_plan(BOW_X_sorted[i], BOW_X_sorted[j])
            sfw_dist_matrix[i,j] =  torch.dot( torch.pow( dist_poincare(X_sorted[i][GI[:,1].astype(int)],X_sorted[j][GI[:,2].astype(int)]), 2), torch.tensor(GI[:,0]).type(torch.FloatTensor).to(device=device))
        
        if i%10==0:
            print(i/M)

    sfw_dmatrix = sfw_dist_matrix+sfw_dist_matrix.T
    np.save('./save_results/'+args.dataset+'/sfw.npy',sfw_dmatrix)
    running_time = time.time() - start_time
    print('Time(SFW) is ', running_time)


else:

    print("Not include such method!")
