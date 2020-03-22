#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        print(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        self.epoch = -1

    # input: N x (L+3)
    def forward(self, input, epoch):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input
        
        if epoch <= 200:
            ldim = 0
            lbase = -100
        elif epoch <= 600:
            ldim = 1
            lbase = 400
        elif epoch <= 1000:
            ldim = 2
            lbase = 800
        else:
            ldim = 3
            lbase = 1200

        for l in range(0, 3):
            lin = getattr(self, "lin" + str(l))
            xi = x
            if l != 0 and self.xyz_in_all:
                xi = torch.cat([xi, xyz], 1)
            x = lin(xi)
            if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm: 
                bn = getattr(self, "bn" + str(l))  
                x = bn(x)
            x = self.relu(x)
            if self.dropout is not None and l in self.dropout:    
                x = F.dropout(x, p=self.dropout_prob, training=self.training)

        for l in range(3, ldim+5):
            lin = getattr(self, "lin" + str(l))
            xi = x
            if l in self.latent_in:
                xi = torch.cat([xi, input], 1)
            elif l != 0 and self.xyz_in_all:
                xi = torch.cat([xi, xyz], 1)
            # add the last layer as residual block
            if l == ldim + 4 and ldim > 0 and lbase - epoch > 0: 
                x = xi * max([0,float(lbase - epoch) / 200]) + lin(xi) * min([1, float(epoch - lbase)/200 + 1])
            else:
                x = lin(xi)
            if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                bn = getattr(self, "bn" + str(l))
                x = bn(x)
            x = self.relu(x)
            if self.dropout is not None and l in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # Last layer
        l = 8
        lin = getattr(self, "lin8")
        xi = x
        if  self.xyz_in_all:
            xi =  torch.cat([xi, xyz], 1)
        x = lin(xi)

        if self.use_tanh:
            x = self.tanh(x)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
