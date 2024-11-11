import torch
import os
import torch.autograd as autograd

def gradient_penalty(critic, real_data, fake_data, device='cpu'):
    # Random weight term for interpolation
    alpha = torch.rand(real_data.size(0), 1).to(device)
    alpha = alpha.expand_as(real_data)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    # Compute critic's prediction on interpolated data
    crit_interpolated = critic(interpolated)


    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=crit_interpolated, inputs=interpolated,
        grad_outputs=torch.ones(real_data.size(0), 1),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    # Compute gradient norm
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # L2 norm penalty
    return penalty

def Earth_Mover_Loss(output, target):
    output_sorted, _ = torch.sort(output)
    target_sorted, _ = torch.sort(target)
    
    # Calculate the cumulative sums
    cdf_output = torch.cumsum(output_sorted, dim=-1)
    cdf_target = torch.cumsum(target_sorted, dim=-1)
    
    # Compute the Earth Mover Loss as the L1 distance between cumulative distributions
    em_loss = torch.mean(torch.abs(cdf_output - cdf_target))
    return em_loss

def D_train(x, G, D, D_optimizer):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    D_real_output = D(x)
    z = torch.randn(x.size(0), 100)
    x_fake = G(z).detach()
    D_fake_output = D(x_fake)
    #D_fake_real_loss = criterion(D_real_output, D_fake_output)
    gp_loss = gradient_penalty(D, x, x_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = -torch.mean(D_real_output) + torch.mean(D_fake_output) + 10*gp_loss
    D_loss.backward()
    D_optimizer.step()

    """for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)"""
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.size(0), 100)
    x_fake = G(z)
    D_output = D(x_fake)
    G_loss = -torch.mean(D_output)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder, suffix):
    torch.save(G.state_dict(), os.path.join(folder,'G' + suffix + '.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D' + suffix + '.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'), map_location=torch.device('cpu'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_discriminator(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'), map_location=torch.device('cpu'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D