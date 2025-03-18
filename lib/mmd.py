import torch


def imq(x, y, h):
    return h/(h+torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1) ** 2)

def mmd(x, y, kernel, h):
    ## Unbiased estimate
    Kxx = kernel(x, x, h)
    Kyy = kernel(y, y, h)
    Kxy = kernel(x, y, h)

    n = x.shape[0]
    cpt1 = (torch.sum(Kxx)-torch.sum(Kxx.diag()))/(n-1) ## remove diag terms
    cpt2 = (torch.sum(Kyy)-torch.sum(Kyy.diag()))/(n-1)
    cpt3 = torch.sum(Kxy)/n

    return (cpt1+cpt2-2*cpt3)/n