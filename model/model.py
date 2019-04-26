import math
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable

from base import BaseModel
from model.loss import make_vae_loss


class AffineVAE(nn.Module):
    def __init__(self, pre_trained_VAE=None, latent_size=16, batch_opt_params=None, 
                 img_size=28, input_dim=1, output_dim=1, 
                 rotation_only=False, use_STN=False):
        """Do we always need the device whenever we're creating tensors in the model? Whats the proper way to do this?"""
        super(AffineVAE, self).__init__()
        if pre_trained_VAE is None:
            self.VAE = VAE(input_dim=input_dim, output_dim=output_dim, latent_size=latent_size, img_size=img_size)
        else:
            self.VAE = pre_trained_VAE
        self.img_size = img_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rotation_only = rotation_only
        self.batch_opt_params = batch_opt_params
        self.use_STN = use_STN
        
        if self.use_STN: #(https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
            if self.rotation_only:
                final_size = 1
                final_bias = torch.cuda.FloatTensor([0])
            else:
                final_size = 2*3
                final_bias = torch.cuda.FloatTensor([1, 0, 0, 0, 1, 0])
            self.localization = nn.Sequential(
                nn.Conv2d(self.input_dim, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.AdaptiveMaxPool2d(output_size=(3, 3)),
                nn.ReLU(True)
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32),
                nn.ReLU(True),
                nn.Linear(32, final_size)
            )
 
            # initialize to identity for rotation or affine
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(final_bias)
            
    def get_stn_params(self, x):
        x = self.localization(x)
        x = x.view(-1, 10 * 3 * 3)
        stn_output = self.fc_loc(x)
        if self.rotation_only:
            affine_params = torch.cuda.FloatTensor(stn_output.size()[0], 2, 3).fill_(0)
            affine_params = self.theta_to_affine(stn_output, affine_params)
        else:
            affine_params = stn_output.view(-1, 2, 3)
        return affine_params

    def affine(self, x, affine_params, padding_mode='zeros'):
        grid = F.affine_grid(affine_params, x.size()).cuda()
        x = F.grid_sample(x, grid, padding_mode=padding_mode)
        return x
    
    def affine_inv(self, x, affine_params, padding_mode='zeros'):
        inv_affine_params = torch.cuda.FloatTensor(affine_params.size()).fill_(0)
        A_inv =  torch.inverse(affine_params[:, :, :2].squeeze())
        b = affine_params[:, : , 2:]
        b_inv = torch.matmul(A_inv, b)
        b_inv = b_inv.squeeze()
        inv_affine_params[:, :2, :2] = A_inv
        inv_affine_params[:, :, 2] = -1*b_inv
        grid = F.affine_grid(inv_affine_params, x.size()).cuda()
        x = F.grid_sample(x, grid, padding_mode=padding_mode)
        return x

    def optimize_rotation(self, x, num_times=100, iterations=50):
        """ SGD on theta, minimizing VAE loss.  
                
        ONLY FOR BS=1. deterministic is True during optimization
        
        num_times: number of random restarts
        iterations: nuber of steps of SGD
        
        returns: loss, affine_params(2x3, not theta)
        """
        
        lr = .01
        best_loss = 1e10
        vae_loss = make_vae_loss(KLD_weight=1)
        
        with torch.enable_grad():
            for trial in range(num_times):
                theta = torch.cuda.FloatTensor(1).uniform_(-2*math.pi, 2*math.pi)
                theta = theta.data.clone().detach().requires_grad_(True).cuda()
                optimizer = optim.Adam([theta], lr=lr)

                for i in range(iterations):
                    affine_params = torch.cat([torch.cos(theta), torch.sin(theta), 
                                       torch.tensor([0.0], requires_grad=True, device="cuda"), 
                                       -1*torch.sin(theta), torch.cos(theta),
                                       torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3)
                    
                    x_affine = self.affine(x, affine_params)
                    mu_logvar = self.VAE.encode(x_affine)
                    z = self.VAE.reparameterize(mu_logvar, deterministic=True)
                    recon_x = self.VAE.decode(z)
                    recon_x = self.affine_inv(recon_x, affine_params)
                    loss = vae_loss((recon_x, mu_logvar), x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # check if optimal every iteration, because why not?
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_affine_params = affine_params.clone().detach()
        
        return best_affine_params, best_loss
    
    def optimize_affine(self, x, num_times=100, iterations=50, translate=False):
        """ SGD on 4 affine parms, minimizing VAE loss.  
                
        ONLY FOR BS=1. deterministic is True during optimization
        
        num_times: number of random restarts
        iterations: nuber of steps of SGD
        
        returns: loss, affine_params(2x3, not theta)
        """
        
        lr = .03
        best_loss = 1e10
        vae_loss = make_vae_loss(KLD_weight=1)
        
        with torch.enable_grad():
            for trial in range(num_times):
                a = torch.cuda.FloatTensor(1).uniform_(-1.5, 1.5).requires_grad_(True).cuda()
                b = torch.cuda.FloatTensor(1).uniform_(-1.5, 1.5).requires_grad_(True).cuda()
                c = torch.cuda.FloatTensor(1).uniform_(-1.5, 1.5).requires_grad_(True).cuda()
                d = torch.cuda.FloatTensor(1).uniform_(-1.5, 1.5).requires_grad_(True).cuda()
                optimizer = optim.Adam([a, b, c, d], lr=lr)

                for i in range(iterations):
                    affine_params = torch.cat([a, b, 
                                       torch.tensor([0.0], requires_grad=True, device="cuda"), 
                                       c, d,
                                       torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3)
                    
                    x_affine = self.affine(x, affine_params)
                    mu_logvar = self.VAE.encode(x_affine)
                    z = self.VAE.reparameterize(mu_logvar, deterministic=True)
                    recon_x = self.VAE.decode(z)
                    recon_x = self.affine_inv(recon_x, affine_params)
                    loss = vae_loss((recon_x, mu_logvar), x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # check if optimal every iteration, because why not?
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_affine_params = affine_params.clone().detach()
        
        return best_affine_params, best_loss
        
        
    def optimize_rotation_batch(self, x, num_times=100, iterations=50, optimize_once=False):
        """ SGD on theta, minimizing VAE loss.  
                
        deterministic is True during optimization
        num_times: number of random restarts
        iterations: nuber of steps of SGD
        optimize_once: Only do SGD on the minumum location
        returns: loss, affine_params(2x3, not theta)
        """
        
        lr = .01
        bs = x.size()[0]
        vae_loss = make_vae_loss(KLD_weight=1)
        
        def vae_loss_unreduced(output, target, KLD_weight=1):
            recon_x, mu_logvar  = output
            mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp(), dim=1)
            BCE = F.mse_loss(recon_x, target, reduction='none')    
            BCE = torch.sum(BCE, dim=(1, 2, 3))
            loss = BCE + KLD_weight*KLD
            return loss    
        
        
        # Find the best rotation for all these samples
        with torch.no_grad():
            theta_loss = torch.cuda.FloatTensor(bs, num_times)
            theta = torch.cuda.FloatTensor(bs, num_times).uniform_(-2*math.pi, 2*math.pi)
            
            for trial in range(num_times):
                affine_params = torch.cat([torch.cat([torch.cos(theta[batch_num, trial]).unsqueeze(0), 
                                                      torch.sin(theta[batch_num, trial]).unsqueeze(0), 
                                                      torch.tensor([0.0], requires_grad=True, device="cuda"), 
                                                      -1*torch.sin(theta[batch_num, trial]).unsqueeze(0), 
                                                      torch.cos(theta[batch_num, trial].unsqueeze(0)),
                                                      torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3) 
                                           for batch_num in range(bs)], dim=0)
                x_affine = self.affine(x, affine_params)
                mu_logvar = self.VAE.encode(x_affine)
                z = self.VAE.reparameterize(mu_logvar, deterministic=True)
                recon_x = self.VAE.decode(z)
                recon_x = self.affine_inv(recon_x, affine_params)
                loss = vae_loss_unreduced((recon_x, mu_logvar), x)                    
                theta_loss[:, trial] = loss.clone().detach()
            min_loss_idx = torch.argmin(theta_loss, dim=1)
            min_theta = theta[torch.arange(theta.size(0)), min_loss_idx]
        
        # now do sgd on these values of theta
        with torch.enable_grad():
            theta = min_theta.squeeze()
            theta = theta.data.clone().detach().requires_grad_(True).cuda()
            optimizer = optim.Adam([theta], lr=lr)
                        
            for i in range(iterations):
                affine_params = torch.cat([torch.cat([torch.cos(theta[batch_num]).unsqueeze(0),
                                      torch.sin(theta[batch_num]).unsqueeze(0), 
                                      torch.tensor([0.0], requires_grad=True, device="cuda"), 
                                      -1*torch.sin(theta[batch_num]).unsqueeze(0), 
                                      torch.cos(theta[batch_num].unsqueeze(0)),
                                      torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3) 
                           for batch_num in range(bs)], dim=0)
                            
                x_affine = self.affine(x, affine_params)
                mu_logvar = self.VAE.encode(x_affine)
                z = self.VAE.reparameterize(mu_logvar, deterministic=True)
                recon_x = self.VAE.decode(z)
                recon_x = self.affine_inv(recon_x, affine_params)
                loss = vae_loss((recon_x, mu_logvar), x)
                # Just in case this SGD is really useless
                if i ==0:
                    best_loss = loss.item()
                    best_affine_params = affine_params.clone().detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # check if optimal every iteration, because why not?
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_affine_params = affine_params.clone().detach()
        return best_affine_params, best_loss
        
      
    def forward(self, x, theta=None, affine_params=None, 
                deterministic=False, return_affine=False):
        """forward pass with optional STN， affine transform, or theta.
        
        Defaults to using batch opt if its specified when model constructed
        Have to recreate model to disable it for normal unoptimized forward pass
        """

        # learned affine 
        if self.use_STN:
            affine_params = self.get_stn_params(x)
            
        # initalize affine to rotation 
        elif theta is not None:
            theta = torch.cuda.FloatTensor([theta])
            affine_params = torch.cat([torch.cos(theta), torch.sin(theta), 
                                           torch.tensor([0.0], requires_grad=True, device="cuda"), 
                                           -1*torch.sin(theta), torch.cos(theta), 
                                           torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3)            
            
        # if this is specified we ignore everything else and optimize rotation for whole batch
        elif self.batch_opt_params is not None:
            affine_params, loss = self.optimize_rotation_batch(x, num_times=self.batch_opt_params['num_times'], 
                                                               iterations=self.batch_opt_params['iterations'])
    
        # initialize to identity for each image if affine param not specified and not stn
        elif affine_params is None:
            affine_params = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3).cuda()
            affine_params = affine_params.expand(x.size()[0], affine_params.size()[0], affine_params.size()[1]).clone()
                    
        x_affine = self.affine(x, affine_params)
        mu_logvar = self.VAE.encode(x_affine)
        z = self.VAE.reparameterize(mu_logvar, deterministic)
        recon_x = self.VAE.decode(z)
        recon_x = self.affine_inv(recon_x, affine_params)
        if return_affine:
            return recon_x, mu_logvar, affine_params, x_affine
        else:
            return recon_x, mu_logvar


        
class VAE(nn.Module):
    """VAE based off https://arxiv.org/pdf/1805.09190v3.pdf
    ??? SHould we use global avg pooling and a 1x1 conv to get mu, sigma? Or even no 1x1, just normal conv.

    should the first fc in deconv be making the output batch*8*7*7???
    """
    def __init__(self, input_dim=1, output_dim=1, latent_size=8, img_size=28):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_size = int(16*(img_size/4)**2)

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_conv4 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2, bias=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, padding=2,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv4 = nn.ConvTranspose2d(16, self.output_dim, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)

        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_bn3 = nn.BatchNorm2d(16)

        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.linear_size)

    def forward(self, x, deterministic=False):
        mu_logvar = self.encode(x)
        z = self.reparameterize(mu_logvar, deterministic)
        recon_x = self.decode(z)
        return recon_x, mu_logvar

    def encode(self, x):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.elu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_conv4(x)
        x = x.view(x.size(0), -1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, 16, int(self.img_size/4), int(self.img_size/4)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.elu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)
        if self.input_dim==1: return torch.sigmoid(x)
        else: return x

    def reparameterize(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic: # return mu 
            return mu
        else: # return mu + random
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

        
# class AffineVAE OLD (nn.Module):
#     def __init__(self, pre_trained_VAE=None, img_size=28, input_dim=1, output_dim=1, latent_size=8, rotation_only=False, use_STN=False):
#         """Do we always need the device whenever we're creating tensors in the model? Whats the proper way to do this?"""
#         super(AffineVAE, self).__init__()
#         if pre_trained_VAE is None:
#             self.VAE = VAE(input_dim=input_dim, output_dim=output_dim, latent_size=latent_size, img_size=img_size)
#         else:
#             self.VAE = pre_trained_VAE
#         self.img_size = img_size
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.rotation_only = rotation_only
#         self.theta = None
#         self.affine_params=None
#         self.optim_params=None
#         self.use_STN = use_STN
        
#         if self.use_STN: #(https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
#             if self.rotation_only:
#                 final_size = 1
#                 final_bias = torch.cuda.FloatTensor([0])
#             else:
#                 final_size = 2*3
#                 final_bias = torch.cuda.FloatTensor([1, 0, 0, 0, 1, 0])
#             self.localization = nn.Sequential(
#                 nn.Conv2d(self.input_dim, 8, kernel_size=7),
#                 nn.MaxPool2d(2, stride=2),
#                 nn.ReLU(True),
#                 nn.Conv2d(8, 10, kernel_size=5),
#                 nn.AdaptiveMaxPool2d(output_size=(3, 3)),
#                 nn.ReLU(True)
#             )
#             self.fc_loc = nn.Sequential(
#                 nn.Linear(10 * 3 * 3, 32),
#                 nn.ReLU(True),
#                 nn.Linear(32, final_size)
#             )
 
#             # initialize to identity for rotation or affine
#             self.fc_loc[2].weight.data.zero_()
#             self.fc_loc[2].bias.data.copy_(final_bias)
            
#     def get_stn_params(self, x):
#         x = self.localization(x)
#         x = x.view(-1, 10 * 3 * 3)
#         stn_output = self.fc_loc(x)
#         if self.rotation_only:
#             affine_params = torch.cuda.FloatTensor(stn_output.size()[0], 2, 3).fill_(0)
#             affine_params = self.theta_to_affine(stn_output, affine_params)
#         else:
#             affine_params = stn_output.view(-1, 2, 3)
#         return affine_params
            
# #     def theta_to_affine(self, theta, affine_params):
# #         affine_params[:, 0, 0] = torch.cos(theta)
# #         affine_params[:, 0, 1] = torch.sin(theta)
# #         affine_params[:, 1, 0] = -1*torch.sin(theta)
# #         affine_params[:, 1, 1] = torch.cos(theta)
# #         return affine_params

#     def affine(self, x, affine_params, padding_mode='zeros'):
#         grid = F.affine_grid(affine_params, x.size()).cuda()
#         x = F.grid_sample(x, grid, padding_mode=padding_mode)
#         return x
    
#     def affine_inv(self, x, affine_params, padding_mode='zeros'):
#         inv_affine_params = torch.cuda.FloatTensor(affine_params.size()).fill_(0)
#         A_inv =  torch.inverse(affine_params[:, :, :2].squeeze())
#         b = affine_params[:, : , 2:]
#         b_inv = torch.matmul(A_inv, b)
#         b_inv = b_inv.squeeze()
#         inv_affine_params[:, :2, :2] = A_inv
#         inv_affine_params[:, :, 2] = -1*b_inv
#         grid = F.affine_grid(inv_affine_params, x.size()).cuda()
#         x = F.grid_sample(x, grid, padding_mode=padding_mode)
#         return x
            

#     def optimize_affine_params(self, x, only_rotation=False, only_shear=False, num_times=100, iterations=50, KLD_weight=1):
#         """ Do SGD on the affine params to minimize loss.  ONLY FOR BS=1
        
#         Shouldnt have self for these parameters, becuase there is one model for multiple imgs/parameters
        
#         deterministic is true during optimization
        
#         num_times: number of random restarts
#         iterations: nuber of steps of SGD
        
#         affine_params is 2x3. Final row is omitted
#         """
#         with torch.autograd.set_detect_anomaly(True):
#             vae_loss = make_vae_loss(KLD_weight=1)

#             lr = .1
#             best_loss = 10000000000000

#             for trial in range(num_times):
#                 if only_rotation:
#                     x_rot = torch.tensor([0.0], dtype=torch.float32,\
#                                          requires_grad=True, device = "cuda")
#                     x_sin = torch.sin(x_rot)
#                     print('x_sin.requires_grad', x_sin.requires_grad)
                    
# #                     print(type(self.VAE.parameters()))
# #                     print(type(list(self.VAE.parameters())[0]))
# #                     theta = nn.Parameter(torch.from_numpy(np.random.uniform(-2*math.pi, 2*math.pi, 1)).float().clone().detach().cuda())
#                     theta = nn.Parameter(torch.cuda.FloatTensor(1).uniform_(-2*math.pi, 2*math.pi))
# #                     theta = torch.tensor(torch.cuda.FloatTensor([0.0]).uniform_(-2*math.pi, 2*math.pi).data, requires_grad=True)
# #                     print(theta)
# #                     print('theta.requires_grad', theta.requires_grad)
# #                     print('theta[0].requires_grad', theta[0].requires_grad)
# #                     add = torch.add(theta, 2.0)
# #                     abc = torch.sin(theta)
# #                     print('abc.requires_grad', abc.requires_grad)
# #                     print('add.requires_grad', add.requires_grad)

#                     affine_params = torch.cat([torch.cos(theta), torch.sin(theta), 
#                                torch.tensor([0.0], requires_grad=False, device="cuda"), 
#                                -1*torch.sin(theta), torch.cos(theta),
#                                torch.tensor([0.0], requires_grad=False, device="cuda")]).view(-1, 2, 3)
                    
#                 elif only_shear:
# #                     c_x = torch.cuda.FloatTensor([0.0]).uniform_(1, 1.5)
#                     c_x = torch.cuda.FloatTensor([0.0]).uniform_(-.3, .3)
#                     c_y = torch.cuda.FloatTensor([0.0]).uniform_(-.3, .3)
# #                     c_y = torch.cuda.FloatTensor([0.0]).uniform_(.6, 1.4)

#                     optim_params = [c_x, c_y]
#                     affine_params = torch.cat([torch.tensor([1.0], requires_grad=True, device="cuda"), c_x, 
#                                torch.tensor([0.0], requires_grad=True, device="cuda"), 
#                                c_y, torch.tensor([1.0], requires_grad=True, device="cuda"), 
#                                torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3)
                                        
#                 else: # initialize to some resonable amount of scaling. This currently includes weird shears.
#                     affine_params = 4*torch.rand(x.size()[0], 2, 3, requires_grad=True) -2
#                     optim_params = [affine_params]
                    
#                 if iterations > 0:
#                     x = nn.Parameter(x)
#                     optimizer = torch.optim.Adam(optim_params, lr=lr)
#                     print('optimizer', optimizer)
#                     for i in range(iterations):
#                         print('x.requires_grad', x.requires_grad)
#                         print('affine_params.requires_grad', affine_params.requires_grad)
#                         x_affine = self.affine(x, affine_params)
#                         print('x_affine.requires_grad', x_affine.requires_grad)
#                         mu_logvar = self.VAE.encode(x_affine)
#                         z = self.VAE.reparameterize(mu_logvar, deterministic=True)
#                         recon_x = self.VAE.decode(z)
#                         recon_x = self.affine_inv(recon_x, affine_params)
#                         print('recon_x.requires_grad', recon_x.requires_grad)
                        
#                         loss = vae_loss((recon_x, mu_logvar), x)
# #                         print(loss.item(), self.theta, self.affine_params[0, 0, 0])
# #                         print(self.VAE.dec_conv1.weight.grad.data.sum())
                        
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()
                        
#                         if loss.item() < best_loss:
#                             best_loss = loss.item()
#                             best_affine_params = affine_params
#                 else:
#                     x_affine = self.affine(x, affine_params)
#                     recon_x, mu_logvar = self.VAE(x_affine, deterministic=True)
#                     recon_x = self.affine_inv(recon_x, affine_params)
#                     loss = vae_loss((recon_x, mu_logvar), x)
#                     if loss.item() < best_loss:
#                         best_loss = loss.item()
#                         best_affine_params = affine_params
#             return best_affine_params, best_loss

        
#     def forward(self, x, theta=None, affine_params=None, deterministic=False, return_affine=False):
#         """forward pass with optionally learned affine transform. 
        
#         !! If no affine passed in will also check if model has some

#         Options:
#         None: This is the identity transform, equivalent to normal VAE

#         Delete::::
#         explicit: will use the affine_params provided, else equivalent to None
#         stn: use learned params. If STN module isn't trained, will give nonsense
#         optimized: optimize affine params to minimize reconstruction loss
#         rot_optimized: optimized, but constrained to rotations.
#         """

#         # learned affine 
#         if self.use_STN: 
#             affine_params = self.get_stn_params(x)
            
#         # initalize affine to rotation 
#         elif theta is not None:
#             theta = torch.cuda.FloatTensor([theta])
#             affine_params = torch.cat([torch.cos(theta), torch.sin(theta), 
#                                            torch.tensor([0.0], requires_grad=True, device="cuda"), 
#                                            -1*torch.sin(theta), torch.cos(theta), 
#                                            torch.tensor([0.0], requires_grad=True, device="cuda")]).view(-1, 2, 3)

#         # initialize to identity for each image if affine param not specified and not stn
#         elif affine_params is None:
#             affine_params = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3).cuda()
#             affine_params = affine_params.expand(x.size()[0], affine_params.size()[0], affine_params.size()[1]).clone()
                    
#         x_affine = self.affine(x, affine_params)
#         mu_logvar = self.VAE.encode(x_affine)
#         z = self.VAE.reparameterize(mu_logvar, deterministic)
#         recon_x = self.VAE.decode(z)
#         recon_x = self.affine_inv(recon_x, affine_params)
#         if return_affine:
#             return recon_x, mu_logvar, affine_params, x_affine
#         else:
#             return recon_x, mu_logvar
