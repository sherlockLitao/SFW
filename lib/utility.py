import numpy as np
from scipy.sparse import coo_matrix
import base
from scipy.stats import ortho_group
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn import metrics
import scipy as sp
from hilbertcurve.hilbertcurve import HilbertCurve
import torch
import torch.nn.functional as F
from utils_hyperbolic import *
from horoProj import project_kd

MIN_NORM = 1e-15



#######################################################
def trans(X, k):

    Xs = (X-np.min(X,0))/(np.max(X,0)-np.min(X,0))
    Xs = Xs*(2**k)
    Xs = np.int64(Xs)
    Xs[Xs==(2**k)] = 2**k-1

    return Xs

## equal weights SFW
def SFW(X, Y, hyperbolic_model="Lorentz", spf_curve='Hm_l', k=5, p=2, eps=1e-5):

    """
    Implement Hyperbolic space-filling curve projection Wasserstein distance for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        hyperbolic_model : string, kinds of Hyperbolic model, either "Lorentz" or "Poincare".

        spf_curve : string, kinds of space-filling curve.
            'Hm_l' is hilbert sorting using recursive sort based on Lorentz model;
            'Hk_l' is hilbert sorting using Hilbert indices based on Lorentz model;
            'Mm_l' is morton sorting using recursive sort based on Lorentz model;
            'Hm_p' is hilbert sorting using recursive sort based on Poincare model;
            'Hk_p' is hilbert sorting using Hilbert indices based on Poincare model;
            'Mm_p' is morton sorting using recursive sort based on Poincare model;

        k : int, order of space-filling curve when spf_curve='Hk_l' or 'Hk_p'
        p : int, order of SFW
        eps : float, cutoff in arccosh(\theta), that is, \theta \in [1+eps,\infinity). Since arccosh(\theta) has derivative \infinity as \theta=1, this small constant guarantees a finite derivative.  
        
    Returns
    ----------
        float, Hyperbolic space-filling curve projection Wasserstein distance
    """

    n,d = X.shape

    assert (hyperbolic_model=="Lorentz" or hyperbolic_model=="Poincare"), \
            "Hyperbolic model should only be Lorentz or Poincare"
    assert (spf_curve=="Hm_l" or spf_curve=="Hk_l" or spf_curve=="Mm_l" or spf_curve=="Hm_p" or spf_curve=="Hk_p" or spf_curve=="Mm_p"), \
            "Space-filling curve should only be Hm_l, Hk_l, Mm_l, Hm_p, Hk_p or Mm_p"

    if hyperbolic_model=="Lorentz":
        
        if spf_curve=='Hm_l':
            Xr = base.hilbert_order(X.detach().cpu().numpy())
            Yr = base.hilbert_order(Y.detach().cpu().numpy())

        elif spf_curve=="Hk_l":
            hilbert_curve = HilbertCurve(k, d)
            Xi = trans(X.detach().cpu().numpy(), k)
            Yi = trans(Y.detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_l':
            Xr = base.morton_order(X.detach().cpu().numpy())
            Yr = base.morton_order(Y.detach().cpu().numpy())
        
        elif spf_curve=='Hm_p':    
            Xr = base.hilbert_order(lorentz_to_poincare(X).detach().cpu().numpy())
            Yr = base.hilbert_order(lorentz_to_poincare(Y).detach().cpu().numpy())

        elif spf_curve=="Hk_p":
            hilbert_curve = HilbertCurve(k, d-1)
            Xi = trans(lorentz_to_poincare(X).detach().cpu().numpy(), k)
            Yi = trans(lorentz_to_poincare(Y).detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**((d-1)*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**((d-1)*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_p':
            Xr = base.morton_order(lorentz_to_poincare(X).detach().cpu().numpy())
            Yr = base.morton_order(lorentz_to_poincare(Y).detach().cpu().numpy())

        res = torch.pow( torch.mean( torch.pow( torch.arccosh( torch.clamp(-minkowski_ip(X[Xr,:],Y[Yr,:]), min=1+eps) ), p)), 1/p )        


    else: 

        if spf_curve=='Hm_p':
            Xr = base.hilbert_order(X.detach().cpu().numpy())
            Yr = base.hilbert_order(Y.detach().cpu().numpy())

        elif spf_curve=="Hk_p":
            hilbert_curve = HilbertCurve(k, d)
            Xi = trans(X.detach().cpu().numpy(), k)
            Yi = trans(Y.detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**((d-1)*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**((d-1)*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_p':
            Xr = base.morton_order(X.detach().cpu().numpy())
            Yr = base.morton_order(Y.detach().cpu().numpy())
        
        elif spf_curve=='Hm_l':    
            Xr = base.hilbert_order(poincare_to_lorentz(X).detach().cpu().numpy())
            Yr = base.hilbert_order(poincare_to_lorentz(Y).detach().cpu().numpy())

        elif spf_curve=="Hk_l":
            hilbert_curve = HilbertCurve(k, d+1)
            Xi = trans(poincare_to_lorentz(X).detach().cpu().numpy(), k)
            Yi = trans(poincare_to_lorentz(Y).detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_l':
            Xr = base.morton_order(poincare_to_lorentz(X).detach().cpu().numpy())
            Yr = base.morton_order(poincare_to_lorentz(Y).detach().cpu().numpy())

        res = torch.pow( torch.mean( torch.pow( dist_poincare(X[Xr,:],Y[Yr,:]), p)), 1/p )


    return res





#######################################################
def general_plan(a=None, b=None, dense=False):

    GI = base.general_Plan(
        a.astype(np.float64),
        b.astype(np.float64)).T

    if dense:
        G = coo_matrix(
        (GI[:,0], (GI[:,1], GI[:,2])),
        shape=(a.shape[0], b.shape[0]))

        G = coo_matrix.todense(G)
        G = np.array(G)
        return G
   
    return GI

## unequal weights SFW
def gSFW(X, Y, a, b, is_plan=False, hyperbolic_model="Lorentz", spf_curve='Hm_l', k=3, p=2, eps=1e-5, device="cpu"):

    """
    Implement Hyperbolic space-filling curve projection Wasserstein distance for unequal sample size and unequal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (m, d), samples in the target domain
        a : array-like, shape (n), samples weights in the source domain
        b : array-like, shape (m), samples weights in the target domain
        is_plan : bool, True return transport plan, False return distance
        hyperbolic_model : string, kinds of Hyperbolic model, either "Lorentz" or "Poincare".

        spf_curve : string, kinds of space-filling curve.
            'Hm_l' is hilbert sorting using recursive sort based on Lorentz model;
            'Hk_l' is hilbert sorting using Hilbert indices based on Lorentz model;
            'Mm_l' is morton sorting using recursive sort based on Lorentz model;
            'Hm_p' is hilbert sorting using recursive sort based on Poincare model;
            'Hk_p' is hilbert sorting using Hilbert indices based on Poincare model;
            'Mm_p' is morton sorting using recursive sort based on Poincare model;

        k : int, order of space-filling curve when spf_curve='Hk_l' or 'Hk_p'
        p : int, order of SFW
        eps : float, cutoff in arccosh(\theta), that is, \theta \in [1+eps,\infinity). Since arccosh(\theta) has derivative \infinity as \theta=1, this small constant guarantees a finite derivative.  
        
    Returns
    ----------
        float, Hyperbolic space-filling curve projection Wasserstein distance for unequal sample size and unequal weights
    """

    n,d = X.shape

   

    if hyperbolic_model=="Lorentz":
        
        if spf_curve=='Hm_l':
            Xr = base.hilbert_order(X.detach().cpu().numpy())
            Yr = base.hilbert_order(Y.detach().cpu().numpy())

        elif spf_curve=="Hk_l":
            hilbert_curve = HilbertCurve(k, d)
            Xi = trans(X.detach().cpu().numpy(), k)
            Yi = trans(Y.detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_l':
            Xr = base.morton_order(X.detach().cpu().numpy())
            Yr = base.morton_order(Y.detach().cpu().numpy())
        
        elif spf_curve=='Hm_p':    
            Xr = base.hilbert_order(lorentz_to_poincare(X).detach().cpu().numpy())
            Yr = base.hilbert_order(lorentz_to_poincare(Y).detach().cpu().numpy())

        elif spf_curve=="Hk_p":
            hilbert_curve = HilbertCurve(k, d)
            Xi = trans(lorentz_to_poincare(X).detach().cpu().numpy(), k)
            Yi = trans(lorentz_to_poincare(Y).detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_p':
            Xr = base.morton_order(lorentz_to_poincare(X).detach().cpu().numpy())
            Yr = base.morton_order(lorentz_to_poincare(Y).detach().cpu().numpy())

    else: 

        if spf_curve=='Hm_p':
            Xr = base.hilbert_order(X.detach().cpu().numpy())
            Yr = base.hilbert_order(Y.detach().cpu().numpy())

        elif spf_curve=="Hk_p":
            hilbert_curve = HilbertCurve(k, d)
            Xi = trans(X.detach().cpu().numpy(), k)
            Yi = trans(Y.detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_p':
            Xr = base.morton_order(X.detach().cpu().numpy())
            Yr = base.morton_order(Y.detach().cpu().numpy())
        
        elif spf_curve=='Hm_l':    
            Xr = base.hilbert_order(poincare_to_lorentz(X).detach().cpu().numpy())
            Yr = base.hilbert_order(poincare_to_lorentz(Y).detach().cpu().numpy())

        elif spf_curve=="Hk_l":
            hilbert_curve = HilbertCurve(k, d)
            Xi = trans(poincare_to_lorentz(X).detach().cpu().numpy(), k)
            Yi = trans(poincare_to_lorentz(Y).detach().cpu().numpy(), k)
            Xdistances = np.array(hilbert_curve.distances_from_points(Xi))/(2**(d*k) - 1)
            Xr = np.argsort(Xdistances)
            Ydistances = np.array(hilbert_curve.distances_from_points(Yi))/(2**(d*k) - 1)
            Yr = np.argsort(Ydistances)
        
        elif spf_curve=='Mm_l':
            Xr = base.morton_order(poincare_to_lorentz(X).detach().cpu().numpy())
            Yr = base.morton_order(poincare_to_lorentz(Y).detach().cpu().numpy())



    aa = a.detach().cpu().numpy()[Xr]
    bb = b.detach().cpu().numpy()[Yr]

    if is_plan:
        G = general_plan(aa,bb,dense=True)
        ix = np.argsort(Xr)
        iy = np.argsort(Yr)

        res = G[ix,:]
        res = res[:,iy]
        return res

    else:
        GI = general_plan(aa,bb)

        if hyperbolic_model=="Lorentz":
            res =  torch.dot( torch.pow( torch.arccosh( torch.clamp(-minkowski_ip(X[Xr[GI[:,1].astype(int)]],Y[Yr[GI[:,2].astype(int)]]), min=1+eps)), p)[:,0], torch.tensor(GI[:,0]).type(X.dtype).to(device=device))
        else:
            res =  torch.dot( torch.pow( dist_poincare(X[Xr[GI[:,1].astype(int)]],Y[Yr[GI[:,2].astype(int)]]), p), torch.tensor(GI[:,0]).type(X.dtype).to(device=device))

        return torch.pow(res, 1/p)





## equal weights IPRSFW
def IPRSFW(X, Y, hyperbolic_model="Lorentz", spf_curve='Hm_l', projection_kind="geodesic", q=2, nslice=50, p=2, eps=1e-5, device="cpu"):

    """
    Implement integral projection robust Hyperbolic space-filling curve projection Wasserstein distance for equal sample size and equal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (n, d), samples in the target domain
        hyperbolic_model : string, kinds of Hyperbolic model, either "Lorentz" or "Poincare".

        spf_curve : string, kinds of space-filling curve.
            'Hm_l' is hilbert sorting using recursive sort based on Lorentz model;
            'Mm_l' is morton sorting using recursive sort based on Lorentz model;
            'Hm_p' is hilbert sorting using recursive sort based on Poincare model;
            'Mm_p' is morton sorting using recursive sort based on Poincare model;

        projection_kind : string, kind of hyperbolic projection, either "geodesic" or "horospherical"
        q : int, dimension of subspace
        nslice : int, number of slices 
        p : int, order of SFW
        eps : float, cutoff in arccosh(\theta), that is, \theta \in [1+eps,\infinity). Since arccosh(\theta) has derivative \infinity as \theta=1, this small constant guarantees a finite derivative.  
        device : string, "cpu" or "cuda"
        
    Returns
    ----------
        float, integral projection robust Hyperbolic space-filling curve projection Wasserstein distance for equal sample size and equal weights
    """

    
    n, d = X.shape
    res = 0

    assert (projection_kind=="geodesic" or projection_kind=="horospherical"), \
            "Kind of projection should only be geodesic or horospherical"

    if hyperbolic_model=="Lorentz":
        d = d - 1


    ## to generate less orthogonal matrix
    k = int(q*nslice/d)+1
    projM = np.zeros((d*k,d))
    for j in range(k):
        projM[(j*d):(j*d+d),:] = ortho_group.rvs(dim=d)

    projM = torch.from_numpy(projM).type(X.dtype).to(device)    


    # For horospherical projection, we recommend Poincare model.
    # To apply horospherical projection for Lorentz model, we use function "lorentz_to_poincare".
    # However, horospherical projection could also be directly applied to Lorentz model, see [1].
    
    # For geodesic projection, we recommend Lorentz model.
    # To apply geodesic projection for Poincare model, we use function "poincare_to_lorentz".

    # [1] Chami, Ines, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections." International Conference on Machine Learning. PMLR, 2021.
    if projection_kind == "geodesic":

        if hyperbolic_model=="Lorentz":
            # could be parallel
            for i in range(nslice):
                prox1 = X[:,1:].matmul((projM[(i*q):(i*q+q),:].T))
                prox2 = torch.cat([X[:,[0]],prox1], dim=-1)
                prox2_norm = -minkowski_ip(prox2,prox2)
                proX = prox2/torch.sqrt(prox2_norm.clamp_min(MIN_NORM))

                proy1 = Y[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                proy2 = torch.cat([Y[:,[0]],proy1], dim=-1)
                proy2_norm = -minkowski_ip(proy2,proy2)
                proY = proy2/torch.sqrt(proy2_norm.clamp_min(MIN_NORM))

                res =  res + torch.pow(SFW(proX, proY, hyperbolic_model, spf_curve, p=p, eps=eps), p)

        else:
            XX = poincare_to_lorentz(X)
            YY = poincare_to_lorentz(Y)
            #  could be parallel
            for i in range(nslice):
                prox1 = XX[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                prox2 = torch.cat([XX[:,[0]],prox1], dim=-1)
                prox2_norm = -minkowski_ip(prox2,prox2)
                proX = prox2/torch.sqrt(prox2_norm.clamp_min(MIN_NORM))

                proy1 = YY[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                proy2 = torch.cat([YY[:,[0]],proy1], dim=-1)
                proy2_norm = -minkowski_ip(proy2,proy2)
                proY = proy2/torch.sqrt(proy2_norm.clamp_min(MIN_NORM))

                res =  res + torch.pow(SFW(proX, proY, hyperbolic_model="Lorentz", spf_curve=spf_curve, p=p, eps=eps), p)

    else:

        if hyperbolic_model=="Poincare":
            for i in range(nslice):
                proX = project_kd(projM[(i*q):(i*q+q),:], X, keep_ambient=False)
                proY = project_kd(projM[(i*q):(i*q+q),:], Y, keep_ambient=False)
                res =  res + torch.pow(SFW(proX, proY, hyperbolic_model, spf_curve, p=p, eps=eps), p)

        else:
            XX = lorentz_to_poincare(X)
            YY = lorentz_to_poincare(Y)
            for i in range(nslice):
                proX = project_kd(projM[(i*q):(i*q+q),:], XX, keep_ambient=False)
                proY = project_kd(projM[(i*q):(i*q+q),:], YY, keep_ambient=False)
                res =  res + torch.pow(SFW(proX, proY, hyperbolic_model="Poincare", spf_curve=spf_curve, p=p, eps=eps), p)

    return torch.pow(res/nslice, 1/p)







## unequal weights IPRSFW
def gIPRSFW(X, Y, a, b, hyperbolic_model="Lorentz", spf_curve='Hm_l', projection_kind="geodesic", q=2, nslice=50, p=2, eps=1e-5, device="cpu"):

    """
    Implement integral projection robust Hyperbolic space-filling curve projection Wasserstein distance for unequal sample size and unequal weights

    Parameters
    ----------
        X : array-like, shape (n, d), samples in the source domain
        Y : array-like, shape (m, d), samples in the target domain
        a : array-like, shape (n), samples weights in the source domain
        b : array-like, shape (m), samples weights in the target domain
    
        spf_curve : string, kinds of space-filling curve.
            'Hm_l' is hilbert sorting using recursive sort based on Lorentz model;
            'Mm_l' is morton sorting using recursive sort based on Lorentz model;
            'Hm_p' is hilbert sorting using recursive sort based on Poincare model;
            'Mm_p' is morton sorting using recursive sort based on Poincare model;

        projection_kind : string, kind of hyperbolic projection, either "geodesic" or "horospherical"
        q : int, dimension of subspace
        nslice : int, number of slices 
        p : int, order of SFW
        eps : float, cutoff in arccosh(\theta), that is, \theta \in [1+eps,\infinity). Since arccosh(\theta) has derivative \infinity as \theta=1, this small constant guarantees a finite derivative.  
        device : string, "cpu" or "cuda"
        
    Returns
    ----------
        float, integral projection robust Hyperbolic space-filling curve projection Wasserstein distance for unequal sample size and unequal weights
    """

    
    n, d = X.shape
    res = 0

    assert (projection_kind=="geodesic" or projection_kind=="horospherical"), \
            "Kind of projection should only be geodesic or horospherical"

    if hyperbolic_model=="Lorentz":
        d = d - 1


    ## to generate less orthogonal matrix
    k = int(q*nslice/d)+1
    projM = np.zeros((d*k,d))
    for j in range(k):
        projM[(j*d):(j*d+d),:] = ortho_group.rvs(dim=d)

    projM = torch.from_numpy(projM).type(X.dtype).to(device)    


    if projection_kind == "geodesic":

        if hyperbolic_model=="Lorentz":
            for i in range(nslice):
                prox1 = X[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                prox2 = torch.cat([X[:,[0]],prox1], dim=-1)
                prox2_norm = -minkowski_ip(prox2,prox2)
                proX = prox2/torch.sqrt(prox2_norm.clamp_min(MIN_NORM))

                proy1 = Y[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                proy2 = torch.cat([Y[:,[0]],proy1], dim=-1)
                proy2_norm = -minkowski_ip(proy2,proy2)
                proY = proy2/torch.sqrt(proy2_norm.clamp_min(MIN_NORM))
                res =  res + torch.pow(gSFW(proX, proY, a, b, hyperbolic_model=hyperbolic_model, spf_curve=spf_curve, p=p, eps=eps, device=device), p)

        else:

            XX = poincare_to_lorentz(X)
            YY = poincare_to_lorentz(Y)
            for i in range(nslice):
                prox1 = XX[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                prox2 = torch.cat([XX[:,[0]],prox1], dim=-1)
                prox2_norm = -minkowski_ip(prox2,prox2)
                proX = prox2/torch.sqrt(prox2_norm.clamp_min(MIN_NORM))

                proy1 = YY[:,1:].matmul(projM[(i*q):(i*q+q),:].T)
                proy2 = torch.cat([YY[:,[0]],proy1], dim=-1)
                proy2_norm = -minkowski_ip(proy2,proy2)
                proY = proy2/torch.sqrt(proy2_norm.clamp_min(MIN_NORM))
                res =  res + torch.pow(gSFW(proX, proY, a, b, hyperbolic_model="Lorentz", spf_curve=spf_curve, p=p, eps=eps, device=device), p)

    else:

        if hyperbolic_model=="Poincare":
            for i in range(nslice):
                proX = project_kd(projM[(i*q):(i*q+q),:], X, keep_ambient=False)
                proY = project_kd(projM[(i*q):(i*q+q),:], Y, keep_ambient=False)
                res =  res + torch.pow(gSFW(proX, proY, a, b, hyperbolic_model=hyperbolic_model, spf_curve=spf_curve, p=p, eps=eps, device=device), p)

        else:
            XX = lorentz_to_poincare(X)
            YY = lorentz_to_poincare(Y)
            for i in range(nslice):
                proX = project_kd(projM[(i*q):(i*q+q),:], XX, keep_ambient=False)
                proY = project_kd(projM[(i*q):(i*q+q),:], YY, keep_ambient=False)
                res =  res + torch.pow(gSFW(proX, proY, a, b, hyperbolic_model="Poincare", spf_curve=spf_curve, p=p, eps=eps, device=device), p)


    return torch.pow(res/nslice, 1/p)



