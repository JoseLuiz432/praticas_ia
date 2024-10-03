import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from argparse import Namespace
from .dataset import TSDataset

class Layer(nn.Module):
    '''
    A single fully connected layer with optional batch normalisation and activation.
    '''
    def __init__(self, in_dim, out_dim, bn = True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if bn: layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

    
class Encoder(nn.Module):
    '''
    The encoder part of our VAE. Takes a data sample and returns the mean and the log-variance of the 
    latent vector's distribution.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        self.embeds = nn.ModuleList([
            nn.Embedding(n_cats, emb_size) for (n_cats, emb_size) in self.hparams.embedding_sizes
        ])
        # The input to the first layer is the concatenation of all embedding vectors and continuous
        # values
        in_dim = sum(emb.embedding_dim for emb in self.embeds) + len(self.hparams.cont_vars)
        layer_dims = [in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)],
        )
        self.mu = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        self.logvar = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
    
    def forward(self, x_cont, x_cat):
        x_embed = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]        
        x_embed = torch.cat(x_embed, dim=1)
        x = torch.cat((x_embed, x_cont), dim=1)
        h = self.layers(x)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        return mu_, logvar_, x  # we return the concatenated input vector for use in loss fn
    

class Decoder(nn.Module):
    '''
    The decoder part of our VAE. Takes a latent vector (sampled from the distribution learned by the 
    encoder) and converts it back to a reconstructed data sample.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        hidden_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]
        out_dim = sum(emb_size for _, emb_size in self.hparams.embedding_sizes) + len(self.hparams.cont_vars)
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(hidden_dims[i], hidden_dims[i + 1], bn) for i in range(len(hidden_dims) - 1)],
        )
        self.reconstructed = nn.Linear(hidden_dims[-1], out_dim)
        
    def forward(self, z):
        h = self.layers(z)
        recon = self.reconstructed(h)
        return recon
    
class VAE(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        
    def reparameterize(self, mu, logvar):
        '''
        The reparameterisation trick allows us to backpropagate through the encoder.
        '''
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * self.hparams.stdev
            return eps * std + mu
        else:
            return mu
        
    def forward(self, batch):
        x_cont, x_cat = batch
        assert x_cat.dtype == torch.int64
        mu, logvar, x = self.encoder(x_cont, x_cat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, x
        
    def loss_function(self, obs, recon, mu, logvar):
        recon_loss = F.smooth_l1_loss(recon, obs, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld
                               
    def training_step(self, batch, batch_idx):
        print('Training step')
        self.log('total_loss', torch.tensor(0.0), on_step=True, prog_bar=True, logger=True)
        recon, mu, logvar, x = self.forward(batch)
        # The loss function compares the concatenated input vector including
        # embeddings to the reconstructed vector
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld

        self.log('total_loss', loss.mean(dim=0), on_step=True, prog_bar=True, 
                 logger=True)
        self.log('recon_loss', recon_loss.mean(dim=0), on_step=True, prog_bar=True, 
                 logger=True)
        self.log('kld', kld.mean(dim=0), on_step=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        recon, mu, logvar, x = self.forward(batch)
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld
        self.log('test_loss', loss)
        return loss
        
    def configure_optimizers(self):
        print('Configuring optimizers')
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay, 
                                eps=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1)
        print('Optimizers configured')
        return [opt], [sch]
    
    def train_dataloader(self):
        print('Training data loaded')
        dataset = TSDataset('train', cont_vars=self.hparams.cont_vars, 
            cat_vars = self.hparams.cat_vars, lbl_as_feat=True
        )
        print('Data loaded')
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=1, shuffle=True
        )
    
    def test_dataloader(self):
        dataset = TSDataset('test', cont_vars=self.hparams.cont_vars,
            cat_vars=self.hparams.cat_vars, lbl_as_feat=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=1, 
            pin_memory=True, persistent_workers=True
        )   