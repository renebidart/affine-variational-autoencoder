import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
Both loss and metrics should take arguments in form (output, target)
The only difference between the two is loss is used for backprop, 
metrics is something we care about to look at.

Arguments must be in form (model_forward_output, targets(label, img, etc))
"""


def make_vae_loss(KLD_weight=1):
    def vae_loss(output, target, KLD_weight=KLD_weight):
        """loss is BCE + KLD. target is original x"""
        recon_x, mu_logvar  = output
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        BCE = F.mse_loss(recon_x, target, reduction='sum')
        loss = BCE + KLD_weight*KLD
        return loss
    return vae_loss


