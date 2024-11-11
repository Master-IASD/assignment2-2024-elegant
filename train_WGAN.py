import argparse
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import trange, tqdm
from model import Generator, Discriminator_WGAN_GP
from utils import save_models
from utils_WGP import G_train_WGAN_GP, D_train_WGAN_GP, weights_init, load_model2
from evaluate import compute_fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN-GP.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Size of mini-batches for SGD")
    parser.add_argument("--ckpt_suffix", type=str, default="_unrolled",
                        help="Ckpt suffix. ex : '_unrolled' will store in the folder checkpoints\
                        the files D_unrolled.pth and G_unrolled.pth ")
    parser.add_argument("--pretrained_model_suffix", type=str, default=None)

    args = parser.parse_args()

    verifs_path = "verifs/wgan"
    checkpoints_path = "checkpoints2_wgan"
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs(verifs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Data Pipeline
    print('Loading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(root='data/MNIST/',
                                    train=True,
                                    transform=transform,
                                    download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    print('Dataset Loaded.')
    print('Loading models...')

    mnist_dim = 784
    pretrained = True if args.pretrained_model_suffix is not None else False

    if pretrained :
        G = Generator(g_output_dim=mnist_dim).cuda()
        G = load_model2(G, 'checkpoints_wgan', 'G' + args.pretrained_model_suffix + '.pth')
        G = torch.nn.DataParallel(G).cuda()
        print("Successfully loaded pretrained Generator")

        D = Discriminator_WGAN_GP(d_input_dim=mnist_dim)
        D = load_model2(D, 'checkpoints_wgan', 'D' + args.pretrained_model_suffix + '.pth')
        D = torch.nn.DataParallel(D).cuda()
        print("Successfully loaded pretrained Discriminator")

    else :
        G = Generator(g_output_dim=mnist_dim)
        D = Discriminator_WGAN_GP(d_input_dim=mnist_dim)
        G.apply(weights_init)
        D.apply(weights_init)
        G = torch.nn.DataParallel(G).cuda()
        D = torch.nn.DataParallel(D).cuda()

    print('Models loaded.')

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=1e-3/2, betas=(0.5, 0.999)) # 1e-3 for the first 150 epochs, then /2 for the last 30 epochs (we choose thes)
    D_optimizer = optim.Adam(D.parameters(), lr=3e-4/2, betas=(0.5, 0.999)) # 3e-4 for the first 150 epochs, then /2 for the last 30 epochs

    # Training parameters
    lambda_gp = 10

    print('Start Training with WGAN-GP:')
    for epoch in trange(1, args.epochs + 1, leave=True):
        pbar = tqdm(train_loader)
            
        # Training Discriminator and Generator
        for i, (x, _) in enumerate(pbar):
            x = x.view(-1, mnist_dim).cuda()
            batch_size = x.size(0)

            D_loss = D_train_WGAN_GP(x, G, D, D_optimizer, lambda_gp)
            G_loss = G_train_WGAN_GP(G, D, G_optimizer, batch_size)
            
            pbar.set_postfix(D_loss=D_loss, G_loss = G_loss)

        # Save checkpoints every 10 epochs
        if epoch == 1 or epoch % 10 == 0: # epoch == 1 just to be sure that everything works fine 
            save_models(G, D, checkpoints_path, suffix=args.ckpt_suffix + f"_e{epoch}")
        
        # Save a few images to see how it evolves
        if epoch % 5 == 0:
            n_samples = 0
            z = torch.randn(5, 100).cuda() # Vecteur de bruit aléatoire de taille (100) pour une image. Ici batch_size images
            x = G(z)
            x = x.reshape(5, 28, 28)
            for k in range(x.shape[0]): # Tous les éléments du batch
                torchvision.utils.save_image(x[k:k+1], os.path.join(verifs_path, f'epoch{epoch}_{k}.png'))
