## Base code from https://sites.google.com/view/hyperbolic-tutorial-eccv22/homepage

import warnings
import matplotlib
import torch
import torchvision
import geoopt
import argparse
import sys

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
from tqdm.auto import trange
# from torch.utils.tensorboard import SummaryWriter ## Tensorboard

import resnet
from wrapped_normal_distribution.wrapped_normal import HyperboloidWrappedNormal
from wrapped_normal_distribution.utils_mixture import MixtureSameFamily

from dataset import *
from prototypes import get_prototypes
from loss import Busemann

sys.path.append("../lib")
from utils_hyperbolic import poincare_to_lorentz, lorentz_to_poincare, dist_poincare2
from hhsw import horo_hyper_sliced_wasserstein_poincare
from hsw import hyper_sliced_wasserstein
from sw import sliced_wasserstein
from utility import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="hhsw_mixt", help="Which loss to use")
parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset to use")
parser.add_argument("--dims", type=int, default=2, help="Dimension B^d")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
parser.add_argument("--lambd", type=float, default=1, help="Lambda")
parser.add_argument("--mult", type=float, default=0.75, help="Penalty term")
parser.add_argument("--prop", type=float, default=0.75, help="Proportion of prototype")
parser.add_argument("--scale_var", type=float, default=0.1, help="Scale of variance")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:3")
print("Using device", device, flush=True)


# HYPERPARAMETER
## Number of classes to learn prototypes for 
if args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "cifar100":
    num_classes = 100
    
## dimension of the learned prototypes (d in the equation)
dims = args.dims

prototypes = get_prototypes(dims, num_classes)
# prototypes = torch.from_numpy(np.load("./prototypes/prototypes-"+str(args.dims)+"d-"+str(num_classes)+"c.npy")).float()



### Load data
basedir = "~/torch_datasets/"
batch_size = args.batch_size # 100
kwargs = {"num_workers": 32, "pin_memory":True}

if args.dataset == "cifar10":
    trainloader, testloader = load_cifar10(basedir, batch_size, kwargs)
elif args.dataset == "cifar100":
    trainloader, testloader = load_cifar100(basedir, batch_size, kwargs)


# Hyperbolic hyperparameters
c = 1.0
mult = args.mult # 0.75

    
# General hyperparameters
learning_rate = 0.0005
drop1 = 1000
drop2 = 1100
do_decay = True

# if args.dataset == "cifar10" or args.dataset == "cifar100":
decay = 0.00005
epochs = 1110 # 20 # 100 # 1110


prop = args.prop

# if args.dataset == "cifar10" or args.dataset == "cifar100":
n_mixture = batch_size



if __name__=="__main__":
    # if args.loss == "hhsw_mixt" or args.loss == "hsw_mixt" or args.loss == "swp_mixt" or args.loss == "swl_mixt":
    #     writer = SummaryWriter('runs/classif_busemann_'+args.dataset+"_"+args.loss+"_bs_"+str(batch_size)+"_lamd_"+args.lambd + "_prop_" + args.prop+"_var_"+args.scale_var)
    # else:
    #     writer = SummaryWriter('runs/classif_busemann_'+args.dataset+"_"+args.loss+"_bs_"+str(batch_size))    
    
    # if args.dataset == "cifar10" or args.dataset == "cifar100":
    model = resnet.ResNet(32, dims, 1, prototypes.float(), device)
        
    model = model.to(device)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    manifold = geoopt.PoincareBall(c=c)
    
    print('Training started:')

    lambd = args.lambd
    mu0 = torch.zeros((dims+1,), device=device)
    mu0[0] = 1
    sigma0 = torch.eye(dims, device=device)
    
    ## Mixture
    K = len(prototypes)
    pi = torch.ones((K,), device=device)/K
    mix = D.Categorical(pi)
    mus = prop * prototypes.to(device)
    log_var = torch.zeros((K,), device=device)
    radius = torch.tensor([1.0], device=device)
    wnd = HyperboloidWrappedNormal(radius, poincare_to_lorentz(mus), args.scale_var*torch.exp(log_var).reshape(-1, 1))
    gmm = MixtureSameFamily(mix, wnd)
    
    if args.pbar:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)
        
    # if args.loss == "hhsw_mixt" or args.loss == "hsw_mixt" or args.loss =="swp_mixt" or args.loss == "swl_mixt":
    f_loss = Busemann(dims, mult)

    for i in pbar:
        # Learning rate decay.
        if i in [drop1, drop2] and do_decay:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        avg_loss = 0
        sum_loss = 0
        sum_sw = 0
        sum_bp = 0
        count = 0
        acc = 0
        model.train()        
        
        for bidx, (data, target) in enumerate(trainloader):
            target_tmp = target.to(device)
            target = model.polars[target] 

            data = torch.autograd.Variable(data).to(device)
            target = torch.autograd.Variable(target).to(device)

            output = model(data)
            output_exp_map = manifold.expmap0(output)

            bp = f_loss(output_exp_map, target)
            
            if args.loss == "hhsw_mixt":
                x0 = lorentz_to_poincare(gmm.sample((n_mixture,)))
                sw = horo_hyper_sliced_wasserstein_poincare(output_exp_map, x0, 1000, device=device, p=2)
            elif args.loss == "hsw_mixt":
                x0 = gmm.sample((n_mixture,))
                sw = hyper_sliced_wasserstein(poincare_to_lorentz(output_exp_map), x0, 1000, device=device, p=2)
            # elif args.loss == "swp_mixt":
            #     x0 = lorentz_to_poincare(gmm.sample((n_mixture,))).type(torch.float32)
            #     sw = sliced_wasserstein(output_exp_map, x0, 1000, device=device, p=2)
            # elif args.loss == "swl_mixt":
            #     x0 = gmm.sample((n_mixture,)).type(torch.float32)
            #     sw = sliced_wasserstein(poincare_to_lorentz(output_exp_map), x0, 1000, device=device, p=2)
            # else:
            #     sw = torch.tensor([0], device=device)
            elif args.loss == "SFW_mixt":
                x0 = lorentz_to_poincare(gmm.sample((n_mixture,)))
                sw = SFW(output_exp_map, x0, hyperbolic_model="Poincare", spf_curve='Hm_p')
            elif args.loss == "IPRSFW(geo)_mixt":
                x0 = gmm.sample((n_mixture,))
                sw = IPRSFW(poincare_to_lorentz(output_exp_map), x0, hyperbolic_model="Lorentz", spf_curve='Hm_l', projection_kind="geodesic", q=2, nslice=50, p=2, eps=1e-5, device=device)
            elif args.loss == "IPRSFW(horo)_mixt":
                x0 = lorentz_to_poincare(gmm.sample((n_mixture,)))
                sw = IPRSFW(output_exp_map, x0, hyperbolic_model="Poincare", spf_curve='Hm_p', projection_kind="horospherical", q=2, nslice=50, p=2, eps=1e-5, device=device)


        
            # if args.loss == "hhsw_mixt" or args.loss == "hsw_mixt" or args.loss =="swp_mixt" or args.loss == "swl_mixt":
            loss_func = bp + lambd * sw
                

            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

            sum_loss += loss_func.item()
            sum_sw += sw.item()
            sum_bp += bp.item()
            count += 1.

            output = model.predict(output_exp_map).float()
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(target_tmp.view_as(pred)).sum().item()
            
        avg_loss = sum_loss / float(len(trainloader.dataset))
        avg_acc = acc / float(len(trainloader.dataset))
        avg_sw = sum_sw / float(len(trainloader.dataset))
        avg_bp = sum_bp / float(len(trainloader.dataset))
        
        # writer.add_scalar("Loss", avg_loss, i)
        # writer.add_scalar("Training Accuracy", avg_acc, i)
        # writer.add_scalar("SW", avg_sw, i)
        # writer.add_scalar("Busemann", avg_bp, i)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (i % 10 == 0 or i == epochs - 1):
            valid_acc = 0
            valid_loss = 0
            model.eval()

            with torch.no_grad():
                for data_val, target_val in testloader:
                    data_val = torch.autograd.Variable(data_val).to(device)
                    target_val = torch.autograd.Variable(target_val).to(device)
                    target_loss_val = model.polars[target_val].to(device)

                    output_val = model(data_val).float()
                    output_val_exp_map = manifold.expmap0(output_val)

                    output_val = model.predict(output_val_exp_map).float()
                    pred_val = output_val.max(1, keepdim=True)[1]
                    valid_acc += pred_val.eq(target_val.view_as(pred_val)).sum().item()

                    bp = f_loss(output_val_exp_map, target_loss_val)

                    if args.loss == "hhsw_mixt":
                        x0 = lorentz_to_poincare(gmm.sample((data.shape[0],)))
                        sw = horo_hyper_sliced_wasserstein_poincare(output_val_exp_map, x0, 1000, device=device, p=2)
                    elif args.loss == "hsw_mixt":
                        x0 = gmm.sample((data.shape[0],))
                        sw = hyper_sliced_wasserstein(poincare_to_lorentz(output_val_exp_map), x0, 1000, device=device, p=2)
                    # elif args.loss == "swp_mixt":
                    #     x0 = lorentz_to_poincare(gmm.sample((data.shape[0],))).type(torch.float32)
                    #     sw = sliced_wasserstein(output_val_exp_map, x0, 1000, device=device, p=2)
                    # elif args.loss == "swl_mixt":
                    #     x0 = gmm.sample((data.shape[0],)).type(torch.float32)
                    #     sw = sliced_wasserstein(poincare_to_lorentz(output_val_exp_map), x0, 1000, device=device, p=2)
                    elif args.loss == "SFW_mixt":
                        x0 = lorentz_to_poincare(gmm.sample((n_mixture,)))
                        sw = SFW(output_exp_map, x0, hyperbolic_model="Poincare", spf_curve='Hm_p')
                    elif args.loss == "IPRSFW(geo)_mixt":
                        x0 = gmm.sample((n_mixture,))
                        sw = IPRSFW(poincare_to_lorentz(output_exp_map), x0, hyperbolic_model="Lorentz", spf_curve='Hm_l', projection_kind="geodesic", q=2, nslice=50, p=2, eps=1e-5, device=device)
                    elif args.loss == "IPRSFW(horo)_mixt":
                        x0 = lorentz_to_poincare(gmm.sample((n_mixture,)))
                        sw = IPRSFW(output_exp_map, x0, hyperbolic_model="Poincare", spf_curve='Hm_p', projection_kind="horospherical", q=2, nslice=50, p=2, eps=1e-5, device=device)

                        
                    # if args.loss == "hhsw_mixt" or args.loss == "hsw_mixt" or args.loss == "swp_mixt" or args.loss == "swl_mixt":
                    valid_loss += (lambd*sw+bp).item()


            len_test = len(testloader.dataset)

            valid_avg_acc = valid_acc / float(len_test)
            valid_avg_loss = valid_loss / float(len_test)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Epoch '+str(i)+' :' )
            print('Training Loss:' + str(avg_loss)+' , Training Accuracy: '+str(avg_acc))
            print('Val Loss:' + str(valid_avg_loss)+' , Val Accuracy: '+str(valid_avg_acc), flush=True) 
            
            # writer.add_scalar("Validation Loss", valid_avg_loss, i)
            # writer.add_scalar("Validation Accuracy", valid_avg_acc, i)
            # writer.flush()

   
    # if args.loss == "hhsw_mixt" or args.loss == "hsw_mixt" or args.loss=="swp_mixt" or args.loss=="swl_mixt":
    torch.save(model.state_dict(), "./weights/resnet_"+args.dataset+"_"+args.loss+"_bs_"+str(batch_size)+"_"+str(args.dims)+"_lamd_"+str(args.lambd) + "_prop_" + str(args.prop)+"_var_"+str(args.scale_var)+".model")
