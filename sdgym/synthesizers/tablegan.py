#!/usr/bin/env python
# coding: utf-8
import os
from comet_ml import Experiment

import numpy as np
import torch
from torch.nn import (
    BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, ReLU, Sequential, Sigmoid, Tanh, init)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# from sdgym.synthesizers.base import SynthesizerBase
from sdgym.synthesizers.utils import CATEGORICAL, TableganTransformer


class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input_):
        return self.seq(input_)


class Classifier(Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, numChannels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (numChannels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    layers_G += [Tanh()]

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class TableganSynthesizer:
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 random_dim=100,
                 numChannels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 store_epoch=[300]):

        self.random_dim = random_dim
        self.numChannels = numChannels
        self.l2scale = l2scale

        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data, experiment=None):
        if experiment is not None:
            experiment.log_parameter('batch_size', self.batch_size)
            experiment.log_parameter('random_dim', self.random_dim)
            experiment.log_parameter('numChannels', self.numChannels)
            experiment.log_parameter('GAN version', 'TableGAN')
        
        self.transformer = TableganTransformer(self.meta, self.side)
        train_data = self.transformer.transform(train_data)
        train_data = torch.from_numpy(train_data.astype('float32')).to(self.device)
        dataset = TensorDataset(train_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim, self.numChannels)

        generator = Generator(self.meta, self.side, layers_G).to(self.device)
        discriminator = Discriminator(self.meta, self.side, layers_D).to(self.device)
        classifier = Classifier(self.meta, self.side, layers_C, self.device).to(self.device)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(classifier.parameters(), **optimizer_params)

        generator.apply(weights_init)
        discriminator.apply(weights_init)
        classifier.apply(weights_init)

        max_epoch = max(self.store_epoch)

        for i in range(max_epoch):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = generator(noise)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = generator(noise)
                optimizerG.zero_grad()
                y_fake = discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = generator(noise)
                if classifier.valid:
                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fake)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

                if((id_ + 1) % 100 == 0):
                    print("epoch", i + 1, "step", id_ + 1, loss_d, loss_g, loss_c)

                if experiment is not None:
                    experiment.log_metric('Discriminator Loss', loss_d)
                    experiment.log_metric('Generator Loss', loss_g)
                    if loss_c is not None:
                        experiment.log_metric('Classifier Loss', loss_c)
                        

                
            if i + 1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                }, "{}/model_{}.tar".format(self.working_dir, i + 1))

    def generate(self, n):
        _, self.layers_G, _ = determine_layers(self.side, self.random_dim, self.numChannels)
        generator = Generator(self.meta, self.side, self.layers_G).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])

            generator.eval()
            generator.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = generator(noise)
                data.append(fake.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)
            data = self.transformer.inverse_transform(data[:n])
            ret.append((epoch, data))

        return ret

    def init(self, meta, working_dir):
        self.meta = meta
        self.working_dir = working_dir

        try:
            os.mkdir(working_dir)
        except Exception:
            pass

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= len(self.meta):
                self.side = i
                break
        # figure out image size
