import torch
import os
from torch import nn
from tqdm import tqdm

def Gaussian_Likelihood(e, simplex) :
    e_center = (e.unsqueeze(dim=1) - simplex.mu.unsqueeze(dim=0)).unsqueeze(dim=-1)
    exp_value = torch.exp(-0.5 * torch.matmul(torch.matmul(e_center.transpose(-1, -2), simplex.sigma_inv), e_center))
    sigma_det_rsqrt = simplex.sigma_det_rsqrt.reshape(1, -1, 1, 1)
    w = simplex.w.reshape(1, -1, 1, 1)
    likelihood = (w * sigma_det_rsqrt * exp_value).sum(dim=1).reshape(-1) # Here we do the sum over the number of components
    return likelihood

def D_train_MD(x, G, D, D_optimizer, Lambda, epsilon, simplex):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real) # e in the Paper
    real_likelihood = Gaussian_Likelihood(D_output, simplex)
    D_real_loss = -torch.log(epsilon + real_likelihood).mean()
    # D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda() 

    D_output =  D(x_fake)
    fake_likelihood = Gaussian_Likelihood(D_output, simplex)
    # print(fake_likelihood)
    D_fake_loss = -torch.log(epsilon + Lambda - fake_likelihood).mean()
    
    # D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output
    # print(D_fake_loss)
    # print(D_real_loss)

    # gradient backprop & optimize ONLY D's parameters
    # D_fake_loss.backward()
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train_MD(x, G, D, G_optimizer, Lambda, epsilon, simplex):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    # print(G_output)
    D_output = D(G_output)
    # print(D_output[:,1])
    G_likelihood = Gaussian_Likelihood(D_output, simplex)
    G_loss = torch.log(epsilon + Lambda - G_likelihood).mean()
    # print(G_loss)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


# Copy-pasted from the https://github.com/haihabi/MD-GAN

class ScaledSigmoid(nn.Module):
    def __init__(self, scale: float = 5.0, shift: float = 2.5):
        """
        The Scaled Sigmoid Module
        :param scale: a float scale value
        :param shift: a float shift value
        """
        super(ScaledSigmoid, self).__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.scale * torch.sigmoid(x) - self.shift

class LambdaNetwork(nn.Module):
    def __init__(self, gmm_dim):
        super(LambdaNetwork, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(1, gmm_dim), ScaledSigmoid())

    def forward(self, x):
        x = x.reshape(-1, 1)
        return self.feed_forward(x)

# Slightly changed :

def compute_Lambda(lambda_net, lambda_training_data, optimizer_lambda, nb_iterations, epsilon, gamma, simplex) :
    for i in tqdm(range(nb_iterations)):
        optimizer_lambda.zero_grad()
        e = lambda_net(lambda_training_data)
        lambda_lk = Gaussian_Likelihood(e, simplex)
        lambda_loss = -torch.log(epsilon + lambda_lk).mean()
        if i % 1000 == 0 and i > 0:
            print("Lambda Loss:" + str(lambda_loss.item()))
            for group in optimizer_lambda.param_groups:
                group['lr'] = group['lr'] * gamma
        lambda_loss.backward()
        optimizer_lambda.step()
    e = lambda_net(lambda_training_data)
    lambda_value = Gaussian_Likelihood(e, simplex).sum().item()
    return lambda_value