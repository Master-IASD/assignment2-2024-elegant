import torch
import os
import torch.autograd as autograd

def gradient_penalty(critic, real_data, fake_data, device='cpu'):
    # Random weight term for interpolation
    alpha = torch.rand(real_data.size(0), 1).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    # Compute critic's prediction on interpolated data
    crit_interpolated = critic(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=crit_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(crit_interpolated),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    # Compute gradient norm
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()  # L2 norm penalty
    return penalty


def D_train(x_real, x_fake, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    """x_real, y_real = x, torch.ones(x.shape[0], 1)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1)

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output"""
    D_real_output = D(x_real)
    D_fake_output = D(x_fake)
    #D_fake_real_loss = criterion(D_real_output, D_fake_output)
    gp_loss = gradient_penalty(D, x_real, x_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = -torch.mean(D_real_output) + torch.mean(D_fake_output) + 10*gp_loss
    D_loss.backward()
    D_optimizer.step()

    """for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)"""
        
    return  D_loss.data.item()


def G_train(x_real, x_fake, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()
                
    D_output = D(x_fake)
    G_loss = -torch.mean(D_output)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'), map_location=torch.device('cpu'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
