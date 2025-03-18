import torch
import ot

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.auto import trange

import sys
sys.path.append("..")

from distributions import sampleWrappedNormal
from utils_hyperbolic import *
from .fid_score import *
from torchvision.utils import save_image



criterion = nn.BCELoss(reduction='mean')



def compute_FID(model, d_latent, prior, device, real_cpu):
    n = 10000
    d = d_latent
    latent_distr = prior
    
    L = []
    for k in range(1):
        if latent_distr == "wrappednormal":
            mu = torch.zeros(d+1, dtype=torch.float64, device=device)
            mu[0] = 1
            Sigma = torch.eye(d, dtype=torch.float, device=device)/5
            z = lorentz_to_poincare(sampleWrappedNormal(mu, Sigma, n)).type(torch.FloatTensor).to(device)
        elif latent_distr == "normal":
            z = torch.randn(size=(n, d)).to(device)/5
        else:
            print("Add this new distribution")
    

        gen_imgs = torch.zeros((10000,28,28))
        model.eval()
        with torch.no_grad():
            for k in range(10):
                gen_imgs[1000*k:1000*(k+1)] = model.decoder(z[1000*k:1000*(k+1)]).detach().cpu().reshape(-1,28,28)

        fid = evaluate_fid_score(real_cpu.reshape(-1,28,28,1), gen_imgs.reshape(-1,28,28,1), batch_size=50)
    
    return fid


def sampling(model, device, epoch, prefix, nrow=14, dim=2, latent_distr="wrappednormal"):
    model.eval()
    n_samples = int(nrow ** 2)
    with torch.no_grad():

        if latent_distr == "wrappednormal":
            mu = torch.zeros(dim+1, dtype=torch.float64, device=device)
            mu[0] = 1
            Sigma = torch.eye(dim, dtype=torch.float, device=device)/5
            sample = lorentz_to_poincare(sampleWrappedNormal(mu, Sigma, n_samples)).type(torch.FloatTensor)
        elif latent_distr == "normal":
            sample = torch.randn(size=(n_samples, dim))/5
        else:
            print("Add this new distribution")
        
        sample = model.decoder(sample.to(device)).cpu()
        s = int(int(28 * 28) ** 0.5)
        ## change user's name!
        pathname = '{}_{}/{}_samples_{}.png'.format("/home/user/SFW/WAE/results", prefix, prefix, epoch)
        print(pathname)
        save_image(sample.view(n_samples, 1, s, s), pathname, nrow=nrow)
        print('Done!')


def reconstruction(model, test_loader, device, epoch, prefix, nrow=14):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch = model(data)
            if i < int(nrow / 2):
                n = min(data.size(0), nrow)
                if i == 0:
                    comparison = torch.cat([data[:n], recon_batch[:n]])
                else:
                    comparison = torch.cat([comparison, data[:n], recon_batch[:n]])
            else:
                break
        
        ## change user's name!
        save_image(comparison.cpu(),
                   '{}_{}/{}_recon_{}.png'.format("/home/user/SFW/WAE/results", prefix, prefix, epoch), nrow=nrow)








def train(model, optimizer, n_epochs, train_loader, test_loader, 
            ae_loss, device,  real_cpu, latent_distr="wrappednormal", plot_val=False,
            plot_results=False, bar=False, fid_compute=False, prefix='sfw'):
    
    print(latent_distr, flush=True)
    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    losses = []
    val_losses = []
    rec_losses = []
    w_latent_losses = []
    w_losses = []
    fid_scores = []

    loss_val_epoch = 0
    loss_rec_epoch = 0
    cpt_batch = 0
    cpt_w, cpt_w_latent, = 0, 0

    with torch.no_grad():
        for x_val, _ in test_loader:
            x_val = x_val.to(device)

            model.eval()
            zhat = model.encoder(x_val)
            
            yhat = model.decoder(zhat)
            val_l = ae_loss(x_val, yhat, zhat, latent_distr)
            rec_l = criterion(yhat, x_val)  
            loss_val_epoch += val_l.item()
            loss_rec_epoch += rec_l.item()
            cpt_batch += 1
            
            n, d = zhat.shape
            
            if latent_distr == "wrappednormal":
                mu = torch.zeros(d+1, dtype=torch.float64, device=device)
                mu[0] = 1
                Sigma = torch.eye(d, dtype=torch.float, device=device)/5
                target_latent = lorentz_to_poincare(sampleWrappedNormal(mu, Sigma, n)).type(torch.FloatTensor).to(device)
            elif latent_distr == "normal":
                target_latent = torch.randn(size=zhat.size()).to(device)/5
            else:
                print("Add this new distribution")
            
            
            M = dist_poincare2(zhat, target_latent) 
            a = torch.ones(zhat.shape[0]) / zhat.shape[0]
            b = torch.ones(target_latent.shape[0]) / target_latent.shape[0]
            cpt_w_latent += ot.emd2(a, b, M).item()
            
            M = ot.dist(x_val.reshape(n, -1), yhat.reshape(n, -1), metric="sqeuclidean")
            a = torch.ones(x_val.shape[0]) / x_val.shape[0]
            b = torch.ones(yhat.shape[0]) / yhat.shape[0]
            cpt_w += ot.emd2(a, b, M).item()

        if fid_compute:
            fid_score = compute_FID(model, zhat.shape[1], latent_distr, device, real_cpu)   
            fid_scores.append(fid_score)

        val_losses.append(loss_val_epoch/cpt_batch)
        rec_losses.append(loss_rec_epoch/cpt_batch)
        w_latent_losses.append(cpt_w_latent/cpt_batch)
        w_losses.append(cpt_w/cpt_batch)

        if bar and fid_compute:
            pbar.set_postfix_str(f"val_loss = {val_losses[-1]:.3f}, w_loss = {w_losses[-1]:.3f}, w_latent_loss = {w_latent_losses[-1]:.3f}, rec_loss={rec_losses[-1]:.3f},  FID={fid_scores[-1]:.3f}")
        elif bar:
            pbar.set_postfix_str(f"val_loss = {val_losses[-1]:.3f}, w_loss = {w_losses[-1]:.3f}, w_latent_loss = {w_latent_losses[-1]:.3f}, rec_loss={rec_losses[-1]:.3f}")


    
    for e in pbar:
        loss_epoch = 0
        cpt_batch = 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)

            model.train()

            z_hat = model.encoder(x_batch)            
            y_hat = model.decoder(z_hat)

            l = ae_loss(x_batch, y_hat, z_hat, latent_distr)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += l.item()
            cpt_batch += 1

        losses.append(loss_epoch/cpt_batch)


        loss_val_epoch = 0
        loss_rec_epoch = 0
        cpt_batch = 0
        cpt_w, cpt_w_latent, = 0, 0

        with torch.no_grad():
            latent_test = np.zeros((0,2))
            label_test = np.zeros(0)
            for x_val, label_val in test_loader:
                x_val = x_val.to(device)

                model.eval()
                zhat = model.encoder(x_val)
                if model.d==2:
                    latent_test = np.r_[latent_test, zhat.detach().cpu().numpy()]
                    label_test = np.r_[label_test, label_val]
                
                yhat = model.decoder(zhat)
                val_l = ae_loss(x_val, yhat, zhat, latent_distr)
                rec_l = criterion(yhat, x_val)  
                loss_val_epoch += val_l.item()
                loss_rec_epoch += rec_l.item()
                cpt_batch += 1
                
                n, d = zhat.shape
                
                if latent_distr == "wrappednormal":
                    mu = torch.zeros(d+1, dtype=torch.float64, device=device)
                    mu[0] = 1
                    Sigma = torch.eye(d, dtype=torch.float, device=device)/5
                    target_latent = lorentz_to_poincare(sampleWrappedNormal(mu, Sigma, n)).type(torch.FloatTensor).to(device)
                elif latent_distr == "normal":
                    target_latent = torch.randn(size=zhat.size()).to(device)/5
                else:
                    print("Add this new distribution")
                
                
                M = dist_poincare2(zhat, target_latent) 
                a = torch.ones(zhat.shape[0]) / zhat.shape[0]
                b = torch.ones(target_latent.shape[0]) / target_latent.shape[0]
                cpt_w_latent += ot.emd2(a, b, M).item()
                
                M = ot.dist(x_val.reshape(n, -1), yhat.reshape(n, -1), metric="sqeuclidean")
                a = torch.ones(x_val.shape[0]) / x_val.shape[0]
                b = torch.ones(yhat.shape[0]) / yhat.shape[0]
                cpt_w += ot.emd2(a, b, M).item()


            if model.d==2:
                plt.figure(figsize=(10,10))
                plt.scatter(latent_test[:,0],latent_test[:,1],c=label_test*10,cmap=plt.cm.Spectral)
                plt.savefig("./results_"+prefix+"/latent_"+str(e)+".png", format="png", bbox_inches="tight")

            if fid_compute and e%10==0:
                fid_score = compute_FID(model, zhat.shape[1], latent_distr, device, real_cpu)
                fid_scores.append(fid_score)
                sampling(model, device, e, prefix, nrow=14, dim=zhat.shape[1])
                reconstruction(model, test_loader, device, e, prefix, nrow=14)
                 
            val_losses.append(loss_val_epoch/cpt_batch)
            rec_losses.append(loss_rec_epoch/cpt_batch)
            w_latent_losses.append(cpt_w_latent/cpt_batch)
            w_losses.append(cpt_w/cpt_batch)

        
        if bar and fid_compute and e%10==0:
            pbar.set_postfix_str(f"val_loss = {val_losses[-1]:.3f}, w_loss = {w_losses[-1]:.3f}, w_latent_loss = {w_latent_losses[-1]:.3f}, rec_loss={rec_losses[-1]:.3f},  FID={fid_scores[-1]:.3f}")
        elif bar:
            pbar.set_postfix_str(f"val_loss = {val_losses[-1]:.3f}, w_loss = {w_losses[-1]:.3f}, w_latent_loss = {w_latent_losses[-1]:.3f}, rec_loss={rec_losses[-1]:.3f}")

        
    return w_latent_losses, w_losses, losses, val_losses, rec_losses, fid_scores


