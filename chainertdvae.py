import matplotlib.pyplot as plt
import numpy as np
import chainer
from chainer import Variable
from chainer import variable
from chainer import reporter
from chainer import initializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from PIL import Image
from chainer.backends import cuda
import chainerrl.distribution as D
import random


def seq_vis(target, out, num_prev, num_pred, save=None):
    """
    Visualize a sequence of generated images w.r.t the 
    ground truth.

        (target): [seq_len, height, width]
        (out)   : [num_pred, height, width]
    """
    w = target.shape[1]
    h = target.shape[2]

    assert(num_prev + num_pred <= target.shape[0])
    assert(out.shape[0] == num_pred)
    

    columns = num_prev + num_pred
    rows = 2

    fig = plt.figure(figsize=(12, 2))
    out_cnt = 0
    for i in range(1, columns*rows + 1):
        
        if i <= columns:

            img = target[i-1].reshape(28, 28)
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        else:

            if i - columns <= num_prev:
                img = np.zeros((w, h))
                fig.add_subplot(rows, columns, i)
                plt.imshow(img)
            else:
                img = out[out_cnt].reshape(28, 28)
                fig.add_subplot(rows, columns, i)
                plt.imshow(img)
                out_cnt += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        plt.savefig(save)
    plt.close()

def g_vis(mat, save_file=None):
    
    mat = chainer.backends.cuda.to_cpu(mat) * 255.
    if len(mat.shape) == 2:
        # one image
        
        plt.imshow(mat)
        if save_file:
            plt.savefig(save_file)
        plt.show()

    else:
        # batch
        length = mat.shape[0]
        columns = 10
        rows = int(((length + 9) / 10) * 10 / 10)
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, columns*rows + 1):
            img = mat[i - 1].reshape((28, 28))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            if i == length:
                break

        if save_file:
            plt.savefig(save_file)
        plt.close()
 


class TdVae(Chain):
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
         2. L_x is log p^B_2 - log p^P_2 computed 
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
        
    def __init__(self, n_units, dim_z, directory=None):
        super(TdVae, self).__init__(
            
          
            conv1 = L.Convolution2D(1, 20, 9),
            lin1  = L.Linear(20 * 20 * 20, 400),
            encoder = L.LSTM(400, out_size=n_units),
            
            bn = chainer.links.BatchNormalization(n_units),
         
 
            p_b_2_lin = L.Linear(n_units),
            p_b_2_m   = L.Linear(dim_z),
            p_b_2_v   = L.Linear(dim_z),

            q_I_lin   = L.Linear(n_units),
            q_I_m     = L.Linear(dim_z),
            q_I_v     = L.Linear(dim_z),

            p_b_1_lin = L.Linear(n_units),
            p_b_1_m   = L.Linear(dim_z),
            p_b_1_v   = L.Linear(dim_z),
            
            p_p_2_lin = L.Linear(n_units),
            p_p_2_m   = L.Linear(dim_z),
            p_p_2_v   = L.Linear(dim_z),

            p_d_1     = L.Linear( 20 * 20 * 20),
            p_d_2     = L.Linear(784),
            p_d_conv  = L.Deconvolution2D(20, 1, 9),

        )
        self.it = 0
        self.directory = directory
    
    def __call__(self, x, t):
        """
          X: (batch_size, how_many_prev, H, W)
          t: (batch_size, how_many_out, H, W)
        """
        
        #should be passed by __call__
        jmp_idx    = random.randint(1, 4)  #step 3
        start_idx  = random.randint(4, 10)  #step 2
        end_idx = jmp_idx + start_idx
         
        loss = None
    
        #Reshape to (784) from (28, 28)
        new_x = F.reshape(x.astype(np.float32), (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).data          
        new_t = F.reshape(t.astype(np.float32), (t.shape[0], t.shape[1], t.shape[2] * t.shape[3])).data 

        
        batch_size = x.shape[0]
        num_prev = x.shape[1]
        num_pred = t.shape[1]
       
    


        num_prev = x.shape[1]
        num_pred = t.shape[1]
        self.encoder.reset_state()



        """
        ==============================================
                        Training Loop
        ==============================================
        """ 
        for i in range(start_idx):
            inp = t[:, i, :, :].reshape((batch_size, 1, 28, 28)).astype(np.float32)
            self.encoder(self.bn(self.lin1(F.dropout(F.relu(self.conv1(inp))))))

        state_at_t = self.encoder.h
       
        for i in range(start_idx, start_idx + jmp_idx):

            inp = t[:, i, :, :].reshape((batch_size, 1, 28, 28)).astype(np.float32)
            self.encoder(self.bn(self.lin1(F.dropout(F.relu(self.conv1(inp))))))


      
        observation_at_jmp = new_t[:, start_idx + jmp_idx - 1, :] 
        observation_at_start = new_t[:, start_idx, :]
        state_at_jmp = self.encoder.h
     
        """
        Get a sample of z_2_target
        """   
        p_b_2_linear = self.bn(F.relu(F.dropout(self.p_b_2_lin(state_at_jmp))))
        p_b_2_mean   = self.p_b_2_m(p_b_2_linear)
        p_b_2_var    = F.exp(self.p_b_2_v(p_b_2_linear))
        p_b_2        = D.GaussianDistribution(p_b_2_mean, p_b_2_var)
        z_2_target   = p_b_2.sample()

          
        """
        Get a sample of z_1_target
        """
        q_I_input  = F.concat((z_2_target, state_at_jmp, state_at_t)) 
        q_I_linear = self.bn(F.relu(F.dropout(self.q_I_lin(q_I_input))))
        q_I_mean   = self.q_I_m(q_I_linear)
        q_I_var    = F.exp(self.q_I_v(q_I_linear))
        q_I        = D.GaussianDistribution(q_I_mean, q_I_var)
        z_1_target = q_I.sample()


        """
        Get a sample of z_1 (sample state_at_t only)
        """
        def p_b(state):
            p_b_1_linear = self.bn(F.relu(F.dropout(self.p_b_1_lin(state))))
            p_b_1_mean   = self.p_b_1_m(p_b_1_linear)
            p_b_1_var    = F.exp(self.p_b_1_v(p_b_1_linear))
            p_b_1        = D.GaussianDistribution(p_b_1_mean, p_b_1_var)
            z_1          = p_b_1.sample()
            return z_1, p_b_1
     
        z_1, p_b_1 = p_b(state_at_t)

        """
        Get sample of z_2(from z_1)
        """

        def p_p(cur_z): 
            p_p_2_linear = self.bn(F.relu(F.dropout(self.p_p_2_lin(cur_z))))
            p_p_2_mean   = self.p_p_2_m(p_p_2_linear)
            p_p_2_var    = F.exp(self.p_p_2_v(p_p_2_linear))
            p_p_2        = D.GaussianDistribution(p_p_2_mean, p_p_2_var)
            z_2          = p_p_2.sample()
            return z_2, p_p_2
        z_2, p_p_2 = p_p(z_1)

        """
        Get Reconstruction from z_2
        """
        def p_d(z):
            p_d_1_lin = F.relu(F.dropout(self.p_d_1(z)))
            conv_input = F.reshape(p_d_1_lin, (p_d_1_lin.shape[0], 20, 20, 20))
            recon     = F.sigmoid(self.p_d_conv(conv_input))
            return F.reshape(recon, (z.shape[0], 784))
    
        recon = p_d(z_2_target)

        

        """
        ==============================================
                        Losses
        ==============================================
        """
        l_3  = F.mean_squared_error(p_d(z_1), observation_at_start)
        l_4  = F.mean_squared_error(p_d(z_1_target), observation_at_start)
     
        l_x  = F.mean_squared_error(recon, observation_at_jmp)
        l_1  = p_b_2.log_prob(z_2_target) - p_p_2.log_prob(z_2_target)
        #switch?
        l_2  = q_I.kl(p_b_1)
        loss = 0
        #loss = l_x + F.sum(l_1) + F.sum(l_2) + l_3 + l_4
        loss += l_x
        #diverges with the log prob loss in its current form
      #  loss += .01 * F.sum(l_1) / batch_size
        #loss += .01 * F.sum(l_2) / batch_size
        loss += l_3
        loss += l_4
        """
        ==============================================
                        Testing Loop
        ==============================================
        """ 
        test_int = 100
     
        if self.it % test_int == 0: 
        
            num_left = 5
            samples = np.zeros((batch_size, num_left + 1, 784))
            init_state, _ = p_b(state_at_jmp)
            local_recon = p_d(init_state)
            samples[:, 0, :] = chainer.backends.cuda.to_cpu(local_recon.data)
            for i in range( num_left - 1):
                i += 1

                update, _ = p_p(init_state)
                recon = p_d(update)
                samples[:, i, :] = chainer.backends.cuda.to_cpu(recon.data)
                init_state = update
            true_in = chainer.backends.cuda.to_cpu(new_t[:, end_idx, :])
            samples[:, num_left, :] = true_in
            fn = self.directory + str(self.it) + '.png'
            g_vis(samples[4, :, :].reshape((num_left + 1,28, 28)), save_file=fn)

        if (self.it % 4000 == 0) and (self.it != 0):
            #import pdb; pdb.set_trace()
            pass
        reporter.report({'loss':loss, 'KL':F.sum(l_2), 'avg_KL':F.sum(l_2), 'mse':l_x }, self)
        self.it += 1
        return loss        

 


class MovingMnistNetwork(Chain):
    def __init__(self, sz=None, n=None, directory=None):
        super(MovingMnistNetwork, self).__init__(
            encoder = L.LSTM(784, out_size=784),
            decoder = L.LSTM(784, out_size=784),
            lin = L.Linear(784, 784),
        )
        self.it = 0
        self.n = n
        self.directory = directory
        
    def __call__(self, x, t):
        # X: (batch_size, how_many_prev, H, W)
        # t: (batch_size, how_many_out, H, W)

        #Reshape to (784) from (28, 28)
        new_x = F.reshape(x.astype(np.float32), (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).data          
        new_t = F.reshape(t.astype(np.float32), (t.shape[0], t.shape[1], t.shape[2] * t.shape[3])).data 

        record = False
        if self.it % 25 == 0:
            record = True
       

        batch_size = x.shape[0]
        num_prev = x.shape[1]
        num_pred = t.shape[1]
        self.encoder.reset_state()


        #For each of the inputs, run the forward networ
        for i in range(num_prev):
            self.encoder(new_x[:, i, :])

        loss = None
        if record:
            answers = np.zeros((batch_size, num_pred, 784))
        
        #Set the decoder state as teh encoder state
        self.decoder.reset_state()
        self.decoder.h = self.encoder.h
        for i in range(t.shape[1]):
            
            ans = F.relu(self.lin(F.dropout(self.decoder.h)))
            self.decoder(ans)
            cur_loss = F.mean_squared_error(ans, new_t[:, i, :]) 
            loss = cur_loss if loss is None else loss + cur_loss
            if record:
                if self.directory is not None:
                    for j in range(t.shape[0]):
                        candidate = ans[j, :].data.astype(np.int32) 
                        answers[j, i, :] = chainer.backends.cuda.to_cpu(candidate)


        
        if record and self.directory:
            batch_idx = 3
            out = answers[batch_idx] 
            inp = chainer.backends.cuda.to_cpu(x[batch_idx])
            true_out = chainer.backends.cuda.to_cpu(t[batch_idx])
            true_seq = np.append(inp, true_out, axis=0) 
            fn = self.directory + str(self.it) + '.png'
            seq_vis(true_seq, out, num_prev, num_pred, save=fn)

        reporter.report({'loss':loss}, self)
        self.it += 1
        return loss


