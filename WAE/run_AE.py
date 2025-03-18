# nohup python -u run_AE.py --iprsfwae_geo --fid > iprsfwae_geo.log 2>&1 &
# nohup python -u run_AE.py --wae --fid --prior "normal" > wae.log 2>&1 &


import sys

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torchvision
import argparse
import time
import numpy as np
import random
import ot

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


sys.path.append("../lib")

from autoencoder.nn import AE
from autoencoder.train_wae import train
from autoencoder.fid_score import *

from hhsw import *
from utility import *
from utils_hyperbolic import *
from distributions import sampleWrappedNormal
from mmd import mmd, imq


parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.1, help="gamma")
parser.add_argument("--epochs", type=int, default=800, help="number of epochs")

parser.add_argument("--sfwae", help="If true, use SFW-AE", action="store_true")
parser.add_argument("--iprsfwae_geo", help="If true, use IPRSFW(geo)-AE", action="store_true")
parser.add_argument("--iprsfwae_horo", help="If true, use IPRSFW(horo)-AE", action="store_true")

parser.add_argument("--hhswae", help="If true, use HHSW-AE", action="store_true")
parser.add_argument("--swae", help="If true, use SW-AE", action="store_true")
parser.add_argument("--wae", help="If true, use W-AE", action="store_true")

parser.add_argument("--fid", help="If true, compute FID", action="store_true")
parser.add_argument("--prior", type=str, default="wrappednormal", help="Specify prior")
parser.add_argument("--d_latent", type=int, default=10, help="Dimension of the latent space")
parser.add_argument("--p", type=int, default=2, help="Order of SSW")
parser.add_argument("--seed", type=int, default=2023, help="Random seed")
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()#,
                # torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

## load dataset
train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=500, shuffle=True, num_workers=1, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=1
)

real_cpu = torch.zeros((10000,28,28))

cpt = 0
for data, _ in test_loader:
    real_cpu[cpt:cpt+32] = data[:,0]
    cpt += 32



criterion = nn.BCELoss(reduction='mean')
# criterion = nn.MSELoss(reduction='mean')



def ae_loss(x, y, z, latent_distr="wrappednormal"):
    n, d = z.size()

    if latent_distr == "wrappednormal":
        mu = torch.zeros(d+1, dtype=torch.float64, device=device)
        mu[0] = 1
        Sigma = torch.eye(d, dtype=torch.float, device=device)/5
        target_latent = lorentz_to_poincare(sampleWrappedNormal(mu, Sigma, n)).type(torch.FloatTensor).to(device)  
    elif latent_distr == "normal":
        target_latent = torch.randn(size=z.size()).to(device)/5
    else:
        print("Add this new distribution")
    
    
    if args.sfwae :
        loss_latent = SFW(z, target_latent, hyperbolic_model="Poincare", spf_curve='Hm_p', k=5, p=2, eps=1e-5)
    elif args.iprsfwae_geo:
        loss_latent = IPRSFW(z, target_latent, hyperbolic_model="Poincare", spf_curve='Hm_p', projection_kind="geodesic", q=2, nslice=50, p=2, eps=1e-5, device=device)
    elif args.iprsfwae_horo:
        loss_latent = IPRSFW(z, target_latent, hyperbolic_model="Poincare", spf_curve='Hm_p', projection_kind="horospherical", q=2, nslice=50, p=2, eps=1e-5, device=device)
    elif args.hhswae:
        loss_latent = horo_hyper_sliced_wasserstein_poincare(z, target_latent, 500, device)
    elif args.swae:
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        loss_latent = ot.sliced_wasserstein_distance(z, target_latent, n_projections=500, seed=gen)
    elif args.wae:
        h = 2 * d
        loss_latent = mmd(z, target_latent, imq, h)
 
    reconstruction_loss = criterion(y, x)    
    return reconstruction_loss + args.gamma*loss_latent


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # real_cpu = real_cpu.to(device)    

    print("device =", device, flush=True)
    
    d = args.d_latent
    print("d =", d, flush=True)

    L_w_losses = []
    L_w_latent = []
    L_losses = []
    L_val_losses = []
    L_rec_losses = []
    L_fid = []
    is_hyperbolic = True

    if args.sfwae:
        prefix = 'sfw'
    elif args.hhswae:
        prefix = 'hhsw'
    elif args.swae:
        prefix = 'sw'
        is_hyperbolic = False
    elif args.wae:
        prefix = 'mmd'
        is_hyperbolic = False
    elif args.iprsfwae_geo:
        prefix = 'iprsfwgeo'
    elif args.iprsfwae_horo:
        prefix = 'iprsfwhoro'


    print("Start: ")
    model = AE(16, d, poincare_output=is_hyperbolic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    w_latent_losses, w_losses, losses, val_losses, rec_losses, fid_scores = train(model, optimizer, args.epochs, train_loader, test_loader, ae_loss, 
                                                                       device, real_cpu, args.prior, plot_results=False, 
                                                                       bar=True, fid_compute=args.fid, prefix=prefix)
    
    torch.save(model.state_dict(), "./results_"+prefix+"/"+prefix+"ae_"+args.prior+"_d_"+str(d)+".model")

    L_w_losses.append(w_losses)
    L_w_latent.append(w_latent_losses)
    L_losses.append(losses)
    L_val_losses.append(val_losses)
    L_rec_losses.append(rec_losses)
    L_fid.append(fid_scores)
        
       
    L_w_losses = np.array(L_w_losses)
    L_w_latent = np.array(L_w_latent)
    L_val_losses = np.array(L_val_losses)
    L_losses = np.array(L_losses)
    L_rec_losses = np.array(L_rec_losses)
    L_fid = np.array(L_fid)


    np.save("./results_"+prefix+"/L_w_"+args.prior+"_d_"+str(d)+".npy", L_w_losses)
    np.save("./results_"+prefix+"/L_w_latent_"+args.prior+"_d_"+str(d)+".npy", L_w_latent)
    np.save("./results_"+prefix+"/L_val_"+args.prior+"_d_"+str(d)+".npy", L_val_losses)
    np.save("./results_"+prefix+"/L_train_"+args.prior+"_d_"+str(d)+".npy", L_losses)
    np.save("./results_"+prefix+"/L_rec_"+args.prior+"_d_"+str(d)+".npy", L_rec_losses)
    if args.fid:
        np.save("./results_"+prefix+"/L_fid_"+args.prior+"_d_"+str(d)+".npy", L_fid)


        
   

