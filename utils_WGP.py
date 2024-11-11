import torch
import torch.nn as nn
import os

def D_train_WGAN_GP(x, G, D, D_optimizer, lambda_gp):    
    D.zero_grad()    

    x_real = x.cuda()    
    D_output_real = D(x_real)    
    D_real_loss = -torch.mean(D_output_real)   
     
    z = torch.randn(x.size(0), 100).cuda()    
    x_fake = G(z).detach()    
    D_output_fake = D(x_fake)    
    D_fake_loss = torch.mean(D_output_fake)

    gradient_penalty = compute_gradient_penalty(D, x_real, x_fake)    

    D_loss = D_real_loss + D_fake_loss + lambda_gp * gradient_penalty    
    D_loss.backward()    
    D_optimizer.step()    
    return D_loss.item()

def G_train_WGAN_GP(G, D, G_optimizer, batch_size):    
    G.zero_grad()    
    z = torch.randn(batch_size, 100).cuda()    
    x_fake = G(z)    
    D_output_fake = D(x_fake)    
    G_loss = -torch.mean(D_output_fake)    
    G_loss.backward()    
    G_optimizer.step()    
    return G_loss.item() 

def compute_gradient_penalty(D, real_data, fake_data):

    # Random weight term for interpolation
    alpha = torch.rand(real_data.size(0), 1).cuda()
    alpha = alpha.expand_as(real_data)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True)

    # Compute critic's prediction on interpolated data
    d_interpolates = D(interpolates)
    fake = torch.ones(real_data.size(0), 1).cuda()

    # Compute gradients
    gradients = torch.autograd.grad(outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute gradients norm
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.2)  
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def load_model2(G, folder, model_name): # It can also be used for D
    print(f"Loading model with path : {os.path.join(folder, model_name)}")
    ckpt = torch.load(os.path.join(folder, model_name))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G