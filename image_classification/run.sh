#!/bin/bash

# CIFAR10
python train_busemann.py --dataset "cifar10" --loss "hsw_mixt" --dims 2 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "hsw_mixt" --dims 4 --lambd 1 --scale_var 0.1 --prop 0.75

python train_busemann.py --dataset "cifar10" --loss "hhsw_mixt" --dims 2 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "hhsw_mixt" --dims 4 --lambd 1 --scale_var 0.1 --prop 0.75

python train_busemann.py --dataset "cifar10" --loss "SFW_mixt" --dims 2 --lambd 0.1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar10" --loss "SFW_mixt" --dims 4 --lambd 0.1 --scale_var 0.1 --prop 0.75

## nohup ./run.sh > SFW.log 2>&1 &




## CIFAR100
python train_busemann.py --dataset "cifar100" --loss "hsw_mixt" --dims 10 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "hhsw_mixt" --dims 10 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "IPRSFW(geo)_mixt" --dims 10 --lambd 1 --scale_var 0.1 --prop 0.75
python train_busemann.py --dataset "cifar100" --loss "IPRSFW(horo)_mixt" --dims 10 --lambd 1 --scale_var 0.1 --prop 0.75
