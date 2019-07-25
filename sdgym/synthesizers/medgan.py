import os
from comet_ml import Experiment

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Module, Sequential
from tqdm import trange
from torch.nn.functional import cross_entropy, mse_loss, sigmoid
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

# from sdgym.synthesizer_base import SynthesizerBase
from sdgym.synthesizers.utils import GeneralTransformer


class ResidualFC(Module):
    def __init__(self, input_dim, output_dim, activate, bnDecay):
        super(ResidualFC, self).__init__()
        self.seq = Sequential(
            Linear(input_dim, output_dim, bias=False),
            BatchNorm1d(output_dim, momentum=bnDecay),
            activate()
        )

    def forward(self, input):
        residual = self.seq(input)
        return input + residual


class Generator(Module):
    def __init__(self, random_dim, hidden_dim, bnDecay):
        super().__init__()

        dim = random_dim
        seq = []
        for item in list(hidden_dim)[:-1]:
            assert item == dim
            seq += [ResidualFC(dim, dim, nn.ReLU, bnDecay)]
        assert hidden_dim[-1] == dim
        seq += [
            Linear(dim, dim, bias=False),
            BatchNorm1d(dim, momentum=bnDecay),
            nn.ReLU()
        ]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        dim = data_dim * 2
        seq = []
        for item in list(hidden_dim):
            seq += [
                Linear(dim, item),
                nn.ReLU() if item > 1 else nn.Sigmoid()
            ]
            dim = item
        self.seq = Sequential(*seq)

    def forward(self, input):
        mean = input.mean(dim=0, keepdim=True)
        mean = mean.expand_as(input)
        inp = torch.cat((input, mean), dim=1)
        return self.seq(inp)


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.lin_mu = Linear(dim, embedding_dim)
        self.lin_std = Linear(dim, embedding_dim)

    def forward(self, input):
        l = self.seq(input)
        return self.lin_mu(l), self.lin_std(l)


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super().__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [
                Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, org_input, output_info, training=False):
        return self.seq(org_input)
        # if training:
        #     return self.seq(org_input)
        # else:
        #     input = self.seq(org_input)
        #     st = 0
        #     output = []
        #     for item in output_info:
        #         if item[1] == 'sigmoid':
        #             ed = st + item[0]
        #             output.append(sigmoid(input[:, st:ed]))
        #             st = ed
        #         elif item[1] == 'softmax':
        #             ed = st + item[0]
        #             softmax_out = torch.nn.functional.gumbel_softmax(input[:, st:ed])
        #             noise = d = torch.randn(item[0]).to("cuda:0") * 0.05
        #             softmax_smooth = 0.95 * softmax_out + noise * softmax_out
        #             output.append(softmax_smooth)
        #             st = ed
        #     cat = torch.cat(output, dim=1)
        #     return cat.view(org_input.shape[0], np.sum([x[0] for x in output_info]))


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


def weights_init_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.01)


def aeloss(fake, real, output_info, mu, logvar):
    st = 0
    loss = []
    for item in output_info:
        if item[1] == 'sigmoid':
            ed = st + item[0]
            loss.append(mse_loss(sigmoid(fake[:, st:ed]), real[:, st:ed], reduction='sum'))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss.append(cross_entropy(
                fake[:, st:ed], torch.argmax(real[:, st:ed], dim=-1), reduction='sum'))
            st = ed
        else:
            assert 0

    reconstruct_loss = sum(loss) / fake.size()[0]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruct_loss + KLD


class MedganSynthesizer:
    """docstring for IdentitySynthesizer."""
    def __init__(self,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),          # 128 -> 128 -> 128
                 discriminator_dims=(256, 128, 1),   # datadim * 2 -> 256 -> 128 -> 1
                 compress_dims=(),                   # datadim -> embedding_dim
                 decompress_dims=(),                 # embedding_dim -> datadim
                 bnDecay=0.99,
                 l2scale=0.001,
                 pretrain_epoch=200,
                 batch_size=1000,
                 store_epoch=[200]):

        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims

        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.bnDecay = bnDecay
        self.l2scale = l2scale

        self.pretrain_epoch = pretrain_epoch
        self.batch_size = batch_size
        self.store_epoch = store_epoch

    def train(self, train_data, experiment=None):
        if experiment is not None:
            experiment.log_parameter('batch_size', self.batch_size)
            experiment.log_parameter('pretrain_epoch', self.pretrain_epoch)
            experiment.log_parameter('random_dim', self.random_dim)
            experiment.log_parameter('generator_dims', self.generator_dims)
            experiment.log_parameter('discriminator_dims', self.discriminator_dims)
            experiment.log_parameter('GAN version', 'MedGAN')

        self.transformer = GeneralTransformer(self.meta)
        self.transformer.fit(train_data)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        data_dim = self.transformer.output_dim
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        opt = SGD
        optimizerAE = Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            weight_decay=self.l2scale
        )

        print(f'Pretraining encoders')
        for i in trange(self.pretrain_epoch):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                mu, std = encoder(real)
                emb = reparameterize(mu, std)
                rec = decoder(emb, self.transformer.output_info)
                loss = aeloss(rec, real, self.transformer.output_info)
                loss.backward()
                experiment.log_metric('AELoss', loss)
                optimizerAE.step()

        #         test_batch = dataset[:2][0]
        #         print(test_batch)
        #         encoder_out = encoder(test_batch)
        #         print(encoder_out)
        #         print(decoder(encoder_out, self.transformer.output_info))

        generator = Generator(self.random_dim, self.generator_dims, self.bnDecay).to(self.device)
        discriminator = Discriminator(data_dim, self.discriminator_dims).to(self.device)
        generator.apply(weights_init_uniform)
        discriminator.apply(weights_init_uniform)

        optimizerG = opt(
            list(generator.parameters()) + list(decoder.parameters()),
            #             weight_decay=self.l2scale,
            lr=5e-3,
        )
        optimizerD = opt(discriminator.parameters(),
                         #                          weight_decay=self.l2scale,
                         lr=5e-3)
        print(f'Starting training')
        mean = torch.zeros(self.batch_size, self.random_dim, device=self.device)
        std = mean + 1
        max_epoch = max(self.store_epoch)
        for i in range(max_epoch):
            n_d = 2
            n_g = 1
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.normal(mean=mean, std=std)
                emb = generator(noise)
                fake = decoder(emb, self.transformer.output_info, training=True)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                real_loss = -(torch.log(y_real + 1e-4).mean())
                fake_loss = (torch.log(1.0 - y_fake + 1e-4).mean())
                loss_d = real_loss - fake_loss
                loss_d.backward()
                optimizerD.step()

                if i % n_d == 0:
                    for _ in range(n_g):
                        noise = torch.normal(mean=mean, std=std)
                        emb = generator(noise)
                        fake = decoder(emb, self.transformer.output_info)
                        optimizerG.zero_grad()
                        y_fake = discriminator(fake)
                        loss_g = -(torch.log(y_fake + 1e-4).mean())
                        loss_g.backward()
                        optimizerG.step()
                        
                if((id_ + 1) % 10 == 0):
                    print("epoch", i + 1, "step", id_ + 1, loss_d, loss_g)
                    
                    if experiment is not None:
                        experiment.log_metric('Discriminator Loss', loss_d)
                        experiment.log_metric('Generator Loss', loss_g)


            if i + 1 in self.store_epoch:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict()
                }, "{}/model_{}.tar".format(self.working_dir, i + 1))

    def generate(self, n):
        data_dim = self.transformer.output_dim
        generator = Generator(self.random_dim, self.generator_dims, self.bnDecay).to(self.device)
        decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            generator.load_state_dict(checkpoint['generator'])
            decoder.load_state_dict(checkpoint['decoder'])

            generator.eval()
            decoder.eval()

            generator.to(self.device)
            decoder.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self.batch_size, self.random_dim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                emb = generator(noise)
                fake = decoder(emb, self.transformer.output_info)
                fake = torch.sigmoid(fake)
                data.append(fake.detach().cpu().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]
            data = self.transformer.inverse_transform(data)
            ret.append((epoch, data))
        return ret

    def init(self, meta, working_dir):
        self.meta = meta
        self.working_dir = working_dir
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
