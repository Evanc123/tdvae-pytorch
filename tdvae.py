from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datasets
from torch.distributions.normal import Normal

from torch.distributions import MultivariateNormal

def vis(arr, name):
    plt.imshow(arr)
    plt.savefig(name)
    plt.clf()

def gpu_vis(arr, name):
    vis(arr.cpu().numpy().reshape(28, 28), name)

dataset_len = 20000
seq_len = 20
device = torch.device("cpu")
print('Loading Dataset of Length ', dataset_len)
dataset = datasets.make_dataset(dataset_len, seq_len)

dataset = torch.Tensor(dataset).to(device)

bce = torch.nn.BCELoss()

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


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
        self.b_t_size = 400
        self.z_size = 50 #50
        self.batch_size = batch_size


        self.fenc1 = nn.Linear(784, 400)
        self.fenc2 = nn.Linear(400, self.b_t_size)

        """P_b(z | b) """
        #b_t_size
        self.p_b_f_1 = nn.Linear(self.b_t_size, self.b_t_size)
        self.p_b_f_2 = nn.Linear(self.b_t_size, self.b_t_size)
        #self.p_b_f_sig = nn.Linear(self.b_t_size, self.z_size * self.z_size)
        self.p_b_f_sig = nn.Linear(self.b_t_size, self.z_size)
        self.p_b_f_mu  = nn.Linear(self.b_t_size, self.z_size)

        """P(z_2 | z_1) """
        self.p_p_f_1 = nn.Linear(self.z_size, self.b_t_size)
        self.p_p_f_2 = nn.Linear(self.b_t_size, self.b_t_size)
        self.p_p_f_3 = nn.Linear(self.b_t_size, self.b_t_size)
        self.p_p_f_mu = nn.Linear(self.b_t_size, self.z_size)
        #self.p_p_f_sig = nn.Linear(self.b_t_size, self.z_size * self.z_size)
        self.p_p_f_sig = nn.Linear(self.b_t_size, self.z_size)

        """Q(z_t_1 | z_t_2, b_t_1, b_t_2) """
        self.q_I_f_1 = nn.Linear(self.b_t_size + self.b_t_size + self.z_size,
                                 self.b_t_size)
        self.q_I_f_2 = nn.Linear(self.b_t_size, self.b_t_size)
        self.q_I_f_3 = nn.Linear(self.b_t_size, self.b_t_size)
        self.q_I_f_mu = nn.Linear(self.b_t_size, self.z_size)
        #self.q_I_f_sig = nn.Linear(self.b_t_size, self.z_size * self.z_size)
        self.q_I_f_sig = nn.Linear(self.b_t_size, self.z_size)

        """ P_D(x | z)"""
        self.fdec0 = nn.Linear(self.z_size, self.b_t_size)
        self.fdec1 = nn.Linear(self.b_t_size, 400)
        self.fdec2 = nn.Linear(400, 784)

        """ LSTM"""
        self.lstm  = nn.LSTM(self.b_t_size, self.b_t_size, batch_first=True)
        self.eye = torch.eye(self.z_size)#.cuda()
        self.empty_ones = torch.eye(self.z_size).unsqueeze(0).repeat(self.batch_size, 1, 1)#.cuda()

    def p_b(self, b):
        tanh = F.tanh(self.p_b_f_1(b))
        sig  = F.sigmoid(self.p_b_f_2(b))
        out  = tanh * sig
        #out = F.relu(self.p_b_f_2(b))
        mu = self.p_b_f_mu(out)
        sigma = torch.exp(self.p_b_f_sig(out))

        return Normal(mu, sigma)
        #import pdb; pdb.set_trace()
        #eyes = sigma.view(-1, self.z_size, self.z_size) * self.eye
        #dist = MultivariateNormal(mu, eyes)
        #return dist

    def q_I(self, z_t_2, b_t_1, b_t_2):
        pre  = self.q_I_f_1(torch.cat((z_t_2, b_t_1, b_t_2), 1))
        tanh = F.tanh(self.q_I_f_2(pre))
        sig  = F.sigmoid(self.q_I_f_3(pre))
        out  = tanh * sig
        #out = F.relu(self.q_I_f_2(pre))
        #out = F.relu(self.q_I_f_2(pre))
        mu = self.q_I_f_mu(out)
        sigma = torch.exp(self.q_I_f_sig(out))
        return Normal(mu, sigma)
        eyes = sigma.view(-1, self.z_size, self.z_size) * self.eye
        dist = MultivariateNormal(mu, eyes)
        return dist

    def p_p(self, z_1):
        pre = self.p_p_f_1(z_1)
        tanh = F.tanh(self.p_p_f_2(pre))
        sig  = F.sigmoid(self.p_p_f_3(pre))
        out  = tanh * sig
        mu = self.p_p_f_mu(out)
        sigma = torch.exp(self.p_p_f_sig(out))
        return Normal(mu, sigma)
        #eyes = sigma.view(-1, self.z_size, self.z_size) * self.eye
        #dist = MultivariateNormal(mu, eyes)
        #return dist

    def encode(self, x):
        h1 = F.relu(self.fenc1(x))
        return self.fenc2(h1)

    def p_d(self, z):
        h3 = F.relu(self.fdec1(F.relu(self.fdec0(z))))
        return F.sigmoid(self.fdec2(h3))

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.b_t_size).to(device),
                torch.zeros(1, self.batch_size, self.b_t_size).to(device))

    def train(self, dataset, opt):
        for t in range(100000):
            batch_idx = random.randint(0, dataset_len - self.batch_size - 1)
            batch = dataset[batch_idx:batch_idx + batch_size].reshape(self.batch_size, 20, 784)
            batch = Variable(batch, requires_grad=True)
            jmp_idx    = random.randint(1, 1)
            start_idx  = 2
            end_idx = jmp_idx + start_idx
            model = self.lstm
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

            """Get p_b distributions """
            p_b_2 = self.p_b(b_t_2)
            p_b_1 = self.p_b(b_t_1)

            """Get our sample z_t_2 """
            z_t_2 = p_b_2.rsample() #sample()

            """Get the smoothing model """
            q_I = self.q_I(z_t_2, b_t_1, b_t_2)
            z_t_1_q = q_I.rsample() #sample()

            """Initialize forward model """
            p_p = self.p_p(z_t_1_q)

            print('forward p_p max-mean, min-scale', p_p.loc.max(), p_p.scale.min())
            print('z t1q shape', z_t_1_q.shape)

            """Reconstruct x from our belief dist """
            x_rec = self.p_d(z_t_2)#self.p_d(z_t_2)

            """ Losses """
            #l_x = torch.mean(torch.sum((x_rec - batch[:, i].data)**2, dim=1))
            l_x = bce(x_rec, batch[:,i].data)
            #print('l_x', l_x)

            """@ALEX are we sure we can just sum the KL's like this? """
            klt = torch.distributions.kl.kl_divergence(q_I, p_b_1)
            kl_sum = torch.sum(klt, dim=1)
            kl_loss = torch.mean(kl_sum)

            #print('z_t_2', p_b_2.loc, p_b_2.scale)
            print('p_b_2 max-mean, min-stdv', p_b_2.loc.max(), p_b_2.scale.min())

            test_kl = False
            if test_kl:
                def KL(p, q, batch_idx):
                    p_sig = p.scale[batch_idx]
                    p_mu  = p.loc[batch_idx]
                    q_sig = q.scale[batch_idx]
                    q_mu = q.loc[batch_idx]
                    kl = 0.0
                    for i in range(self.z_size):
                        first = torch.log(q_sig[i] / p_sig[i])
                        second = p_sig[i].pow(2) + (p_mu[i] - q_mu[i]).pow(2)
                        third = 2 * q_sig[i].pow(2)
                        kl += first + second / third - .5
                    return kl
                for i in range(self.batch_size):
                    assert(np.isclose(KL(q_I, p_b_1, i).cpu().data.numpy(),
                                      kl_sum[i].cpu().data.numpy()))

            """@ALEX same question for the log-lkelihoods. """
            log_p_b = torch.sum(p_b_2.log_prob(z_t_2) - p_p.log_prob(z_t_2), dim=1)
            l_1 = torch.mean(log_p_b)
            loss = l_x + 0.001 * l_1 + 0.001 * kl_loss

            print("x rec", x_rec.shape, x_rec.max())
            print("x real", batch[:,i].shape, batch[:,i].max())

            if t != 0 and t % 100 == 0:
                """ Visualization help for debugging"""
                print('saving')
                gpu_vis(x_rec.data[0], 'rec.png')
                gpu_vis(batch[0, i].data, 'real.png')
                #import pdb; pdb.set_trace()
            print('mse', l_x.item(), 'logprob', l_1.item(), 'kl', kl_loss, 'step', t)
            #kl -- l2
            loss.backward()
            opt.step()
            opt.zero_grad()

batch_size = 128
mean_field = TdVae(batch_size).to(device)
optimizer = optim.Adam(mean_field.parameters(), lr=.001)
mean_field.train(dataset, optimizer)



