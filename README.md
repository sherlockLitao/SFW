# Efficient Variants of Wasserstein Distance in Hyperbolic Space via Space-filling Curve Projection

This repository is the official implementation of Efficient Variants of Wasserstein Distance in
Hyperbolic Space via Space-filling Curve Projection.
This folder includes main experiments based on SFW, $SFW^\text{IPR}$(geo), and SFW$^\text{IPR}$(horo).


# Platform:
We test these examples in a conda environment on Ubuntu 20.04.4, with cuda 11.6.
All experiments are implemented by a single RTX 3090 GPU.


# Main Dependencies

## python
* argparse, matplotlib, numpy, pickle, pytorch, sklearn, pybind11, hilbertcurve, POT, ...


## C++ library
* eigen3
(install eigen3: sudo apt install libeigen3-dev)




# What is included ?

* folder lib : 
Both C++ based and python based implementations of SFW distance and its invariants.
C++ uses recursive sort and python uses space-filing curve indices.

* folder basic_experiment :
  - convergence.ipynb : SFW and its variants w.r.t. different dimensions
  - different_hyperbolic_model.ipynb : SFW w.r.t. $\mathbb{L}^d$ and $\mathbb{B}^d$ 
  - different_sf_curve.ipynb : SFW w.r.t. Morton and Hilbert curve
  - k_order.ipynb : Analysis of hyperparameter k (order of discrete space-filling curve)
  - Evolution_along_WND.ipynb : Evolution experiments
  - more_on_Evolution_along_WND.ipynb :  Additional experiments of evolution
  - motivation.ipynb : Figure 1 which shows HSW has some disadvantages because it considers projected samples
  - more_on_motivation.ipynb : Additional experiments of Figure 1
  - time.ipynb : Time analysis

* folder Gradient_Flows:
Implementations of gradient flows

* folder BDP :
Implementations of hyperbolic Wasserstein autoencoder for synthetic branching diffusion process(BDP)

* folder WAE :
Implementations of hyperbolic Wasserstein autoencoder for real-data

* folder image_classification :
Image classification for CIFAR10 and CIFAR100

* folder document_classification :
Document classification for 4 dataset
  



# Test our method:

All *.ipynb files provide **figures** in our manuscript.

## Before all, you need to:

Firstly, open a terminal and open file lib/setup.py and change line 7,8.

* line 7 is path of Eigen
* line 8 is path of pybind11
  
Secondly, you can open folder lib and run: 

* python setup.py build_ext --inplace

Then, you can use python to run all code.
Note that there may be some file path problems.

## To be more precise.
For BDP, you can:
open folder BDP and run
* python -u run_BDP.py

For hyperbolic WAE, you can:
open folder WAE and run
* python -u run_AE.py --iprsfwae_geo --fid

For gradient flows, you can:
open folder Gradient_Flows and run
* ./run.sh

For image classification, you can:
open folder image_classification and run
* ./run.sh

For document classification, you need to download the datasets in [https://github.com/mkusner/wmd] and pre-trained model in [https://github.com/alex-tifrea/poincare_glove].
Then, you could open folder document_classification and run
* python -u preprocessing.py --embedding "poincare" --dataset "bbcsport"
* python -u compute_dist_matrix.py --embedding "poincare" --dataset "bbcsport" --method "SFW"
* python -u evaluation.py --dataset "bbcsport"
















# Reference
[https://github.com/CGAL/cgal]

[https://github.com/clbonet/Hyperbolic_Sliced-Wasserstein_via_Geodesic_and_Horospherical_Projections]

[https://github.com/sherlockLitao/HCP]

[https://github.com/HazyResearch/HoroPCA]

[https://github.com/mkusner/wmd]

[https://github.com/alex-tifrea/poincare_glove]

[https://github.com/MinaGhadimiAtigh/Hyperbolic-Busemann-Learning]

[https://github.com/joeybose/HyperbolicNF]
