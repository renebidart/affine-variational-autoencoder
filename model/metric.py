import torch
import torch.nn.functional as F

"""
Return scalars of the metrics we care about to log
So much duplicated computation with loss, but don't want to change interface

??? How to add variable inputs to the loss and metrics?
"""

def BCE(output, target):
    recon_x, mu_logvar = output
    BCE = F.mse_loss(recon_x, target, reduction='sum')
    return BCE.item()

def KLD(output, target):
    recon_x, mu_logvar = output
    mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
    logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
    KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
    return KLD.item()

def vae_loss(output, target, KLD_weight=1):
    """loss is BCE + KLD. target is original x"""
    recon_x, mu_logvar = output

    KLD = KLD(output, target)
    BCE = F.mse_loss(recon_x, target, reduction='sum').item()

    loss = BCE + KLD_weight*KLD
    return loss

