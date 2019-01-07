from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import random
import matplotlib.pyplot as plt
import datasets
from torch.distributions.normal import Normal

from torch.distributions import MultivariateNormal

def vis(arr):
    plt.imshow(arr)
    plt.show()

def gpu_vis(arr):
    vis(arr.cpu().numpy().reshape(28, 28))

dataset_len = 5000
seq_len = 20
device = torch.device("cuda")
print('Loading Dataset of Length ', dataset_len)
dataset = datasets.make_dataset(dataset_len, seq_len)

dataset = torch.Tensor(dataset).to(device)


class TdVae(nn.Module):
    """
       MLP TdVae Implementation:
       3 parts - Training loop, losses, Testing loop
       Training Loop:
         1. Pick random sample
         2. Pick an index
         3. Pick a range in [t, range]
         4. Rollout lstm given x_0 to x_t to x_range,
            accumulating each state
         5. Generate z_2 by feeding s_2 through p^B_2
         6. Generate z_1 through q_1, whcih takes in s_1, s_2
            and z_2
         7. Generate z_1' through p^B_1 by feeding s_1
         8. Generate z_2' through p^P_2 by feeding z_1'
         9. Generate x_2 through p_2^D feeding z_2'

       Losses:
         1. L_x is the reconstruction loss on x_2
         2. L_1 is log p^B_2 - log p^P_2 computed
            using distributions.log_likelihood. This
            term is to make the predictive distribution
            able to compute z_2 from z_1 as if it had
            all the stateful information.
         3. L_2 is the KL divergence betweene q_1 and
            p^B_1. This term tries to force the predictive
            distribution to predict z_1 as if it had access
            to all the information from z_2 and s_2

       Testing Loop:
         1. State s_t is obtained online using the rnn
         2. z_t is generated from s_t, which represent the
            current frame as well as the beliefs about
            future states
         3. Use the decoding distributtion p^D(x_t | z_t) which
            is generally quite close to x_t
         4. Sequentially zmple z <- z' ~ P^P(z' | z) starting
            at z_t
         5. Decode each of the z's
    """
    def __init__(self, batch_size):
        super(TdVae, self).__init__()
        self.rep_size = 50
        self.half_rep = int(self.rep_size / 2)
        self.b_t_size = 50
        self.batch_size = batch_size

        self.fenc1 = nn.Linear(784, 400)
        self.fenc2 = nn.Linear(400, 50)

        """P_b(z | b) """
        self.p_b_f_1 = nn.Linear(50, 50)
        self.p_b_f_2 = nn.Linear(50, 50)
        self.p_b_f_sig = nn.Linear(50, 8)
        self.p_b_f_mu  = nn.Linear(50, 8)

        """P(z_2 | z_1) """
        self.p_p_f_1 = nn.Linear(8, 50)
        self.p_p_f_2 = nn.Linear(50, 50)
        self.p_p_f_3 = nn.Linear(50, 50)
        self.p_p_f_mu = nn.Linear(50, 8)
        self.p_p_f_sig = nn.Linear(50, 8)

        """Q(z_t_1 | z_t_2, b_t_1, b_t_2) """
        self.q_I_f_1 = nn.Linear(50 + 50 + 8, 50)
        self.q_I_f_2 = nn.Linear(50, 50)
        self.q_I_f_3 = nn.Linear(50, 50)
        self.q_I_f_mu = nn.Linear(50, 8)
        self.q_I_f_sig = nn.Linear(50, 8)

        """ P_D(x | z)"""
        self.fdec0 = nn.Linear(8, 50)
        self.fdec1 = nn.Linear(50, 400)
        self.fdec2 = nn.Linear(400, 784)

        """ LSTM"""
        self.lstm  = nn.LSTM(50, self.b_t_size, batch_first=True)

    def p_b(self, b):
        tanh = F.tanh(self.p_b_f_1(b))
        sig  = F.sigmoid(self.p_b_f_2(b))
        out  = tanh * sig
        mu = self.p_b_f_mu(out)
        sigma = torch.exp(self.p_b_f_sig(out))
        dist = Normal(mu, sigma)
        return dist

    def q_I(self, z_t_2, b_t_1, b_t_2):
        pre  = self.q_I_f_1(torch.cat((z_t_2, b_t_1, b_t_2), 1))
        tanh = F.tanh(self.q_I_f_2(pre))
        sig  = F.sigmoid(self.q_I_f_3(pre))
        out  = tanh * sig
        mu = self.q_I_f_mu(out)
        sigma = torch.exp(self.q_I_f_sig(out))
        dist = Normal(mu, sigma)
        return dist

    def p_p(self, z_1):
        pre = self.p_p_f_1(z_1)
        tanh = F.tanh(self.p_p_f_2(pre))
        sig  = F.sigmoid(self.p_p_f_3(pre))
        out  = tanh * sig
        mu = self.p_p_f_mu(out)
        sigma = torch.exp(self.p_p_f_sig(out))
        dist = Normal(mu, sigma)
        return dist

    def encode(self, x):
        h1 = F.relu(self.fenc1(x))
        return self.fenc2(h1)

    def p_d(self, z):
        h3 = F.relu(self.fdec1(F.relu(self.fdec0(z))))
        return self.fdec2(h3)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.b_t_size).to(device),
                torch.zeros(1, self.batch_size, self.b_t_size).to(device))

    def train(self, dataset, opt):
        for t in range(10000):
            batch_idx = random.randint(0, dataset_len - self.batch_size - 1)
            batch = dataset[batch_idx:batch_idx + batch_size].reshape(self.batch_size, 20, 784) / 255.
            jmp_idx    = random.randint(1, 4)
            start_idx  = random.randint(4, 10)
            end_idx = jmp_idx + start_idx

            model = self.lstm
            model.zero_grad()
            model.hidden = self.init_hidden()
            loss = 0.0
            """ Roll Forward LSTM"""
            for i in range(start_idx):
                enc = self.encode(batch[:, i])
                lstm_out, model.hidden = model(enc.unsqueeze(1), model.hidden)
            b_t_1 = lstm_out.squeeze()
            for i in range(start_idx, end_idx):
                enc = self.encode(batch[:, i])
                lstm_out, model.hidden = model(enc.unsqueeze(1), model.hidden)
                b_t_2 = lstm_out.squeeze()

            p_b = self.p_b(b_t_2)
            z_t_2 = p_b.sample()

            q_I = self.q_I(z_t_2, b_t_1, b_t_2)
            z_t_1_q = q_I.sample()

            p_p = self.p_p(z_t_1_q)

            z_t_2_p = p_p.sample()

            x_rec = self.p_d(z_t_2)
            l_x = torch.nn.MSELoss()(x_rec, batch[:, i])
            l_2 = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_I, p_b), dim=1))

            log_p_b = torch.sum(p_b.log_prob(z_t_2_p), dim=1)
            log_p_p = torch.sum(p_p.log_prob(z_t_2_p), dim=1)
            l_1 = torch.mean(log_p_b - log_p_p)
            loss = l_x
            if t == 5000:
                import pdb; pdb.set_trace()
            print('mse', l_x.item(), 'kl', l_2.item(), 'logprob', l_1.item(), 'step', t)
            loss.backward()
            opt.step()

batch_size = 256
mean_field = TdVae(batch_size).to(device)
optimizer = optim.Adam(mean_field.parameters(), lr=1e-5)

mean_field.train(dataset, optimizer)



