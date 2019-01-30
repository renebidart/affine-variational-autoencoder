import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from base import BaseModel
from model.loss import make_vae_loss


class AffineVAE(nn.Module):
    def __init__(self, device, VAE=None, img_size=28, input_dim=1, output_dim=1, latent_size=8, use_STN=False):
        """Do we always need the device whenever we're creating tensors in the model? Whats the proper way to do this?"""
        super(AffineVAE, self).__init__()
        self.device = device
        if VAE:
            self.VAE = VAE
        else:
            self.VAE = VAE(input_dim=input_dim, output_dim=output_dim, latent_size=latent_size, img_size=img_size)
        self.img_size = img_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if use_STN: #(https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
            self.localization = nn.Sequential(
                nn.Conv2d(self.input_dim, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )

            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def affine(self, x, affine_params, padding_mode='zeros'):
        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid, padding_mode=padding_mode)
        return x

    def affine_inv(self, x, affine_params, padding_mode='zeros'):
        # ??? How to do this properly? also need to delete affine_params?
        # can use batch inverse, but then need the batch multiply thing for constant
#         affine_params_inv = affine_params
#         for img_num in range(2):#x.size()[0]):
#             print(affine_params[:2,:,:])
#             img_params = affine_params[img_num, :, :]
#             print('img_params', img_params)
#             A_inv =  torch.inverse(img_params[:, :2].squeeze())
#             img_params[:2, :2] = A_inv
#             img_params[:, 2] = -1*torch.mv(A_inv, img_params[:, 2].squeeze())
#             affine_params_inv[img_num, :, :] = img_params
#             print('affine_params_inv[img_num, :, :]', affine_params_inv[img_num, :, :])
        
    # convert affine to inverse
        for img_num in range(2):#x.size()[0]):
            A_inv =  torch.inverse(affine_params[img_num, :, :2].squeeze())
            affine_params[img_num, :2, :2] = A_inv
            affine_params[img_num, :, 2] = -1*torch.mv(A_inv, affine_params[img_num,: , 2].squeeze())
        grid = F.affine_grid(affine_params, x.size())
        x = F.grid_sample(x, grid, padding_mode=padding_mode)
        return x
                
    def stn_params(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        affine_params = self.fc_loc(xs)
        affine_params = affine_params.view(-1, 2, 3)
        return affine_params
                
    def optimize_affine_params(self, x, only_rotation=False, num_times=100, iterations=50, KLD_weight=1):
        """ Do SGD on the affine params to minimize loss. 
        
        num_times: number of random restarts
        iterations: nuber of steps of SGD
        
        affine_params is 2x3. Final row is omitted
        """
        vae_loss = make_vae_loss(KLD_weight=1)
        
        lr = .001
        results = torch.zeros(x.size()[0], 7) # six params and one loss
        
        affine_params = torch.zeros(x.size()[0], 2, 3)
        best_affine_params = affine_params
        best_loss = 10000000000000
        
        for trial in range(num_times):
            if only_rotation:
                theta = -2*math.pi * torch.rand(x.size()[0]) + 2*math.pi
                affine_params[:, 0, 0] = torch.cos(theta)
                affine_params[:, 0, 1] = torch.sin(theta)
                affine_params[:, 1, 0] = -1*torch.sin(theta)
                affine_params[:, 1, 1] = torch.cos(theta)
                optimizer = optim.Adam([theta], lr=lr)
            else: # initialize to some resonable amount of scaling. This currently includes weird shears.
                affine_params = 4*torch.rand(x.size()[0], 2, 3) -2
                optimizer = optim.Adam([affine_params], lr=lr)
            
#             rand_mu = np.random.normal(0,1, (1, self.latent_size))
#             mu = torch.from_numpy(rand_mu).float().to(self.device)
            
            for i in range(iterations):
                # this is the full forward step of the model using explict affine_params. How to organze this???
                x_affine = self.affine(x, affine_params)
                mu_logvar = self.encode(x_affine)
                z = self.VAE.reparameterize(mu_logvar, deterministic)
                recon_x = self.VAE.decode(z)
                recon_x = self.affine_inv(recon_x, affine_params)
                output = (recon_x, mu, logvar) # put it in format for the loss
                loss = vae_loss(output, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_affine_params = affine_params

        return best_affine_params
        
        
    def forward(self, x, affine_params=None, deterministic=False):
        """forward pass with optionally learned affine transform. 
        
        Options:
        None: This is the identity transform, equivalent to normal VAE
        explicit: will use the affine_params provided, else equivalent to None
        stn: use learned params. If STN module isn't trained, will give nonsense
        optimized: optimize affine params to minimize reconstruction loss
        rot_optimized: optimized, but constrained to rotations.
        """
        
#         assert affine_type in [None, 'explicit', 'stn', 'optimized', 'rot_optimized']
        
        # initialize to identity for each image in batchpyto
        
        if affine_params is None:
            affine_params = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3).to(self.device)
            affine_params = affine_params.expand(x.size()[0], affine_params.size()[0], affine_params.size()[1]).clone()

#         elif affine_type == 'explicit':
#             affine_params = self.affine_params
# #             affine_params = affine_params.expand(x.size()[0], -1, -1)
#         elif affine_type == 'stn':
#             affine_params = stn_get_params(x)
#         elif affine_type == 'optimized':
#             # optimize for each image in the batch individually???
#             # looks like duplicated computation
#             affine_params = self.optimize_affine_params(self, x, num_times=100, iterations=50, KLD_weight=1)
        
        print('affine_params.size() 1', affine_params.size())


        x_affine = self.affine(x, affine_params)
        mu_logvar = self.VAE.encode(x)
        z = self.VAE.reparameterize(mu_logvar, deterministic)
        recon_x = self.VAE.decode(z)
        recon_x = self.affine_inv(recon_x, affine_params)
        return recon_x, mu_logvar, affine_params
    
    
#     def encode_STN(self, x):
#         x, affine_params = self.stn(x)
#         mu_logvar = self.VAE.encode(x)
    

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
