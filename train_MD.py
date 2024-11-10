import torch 
import os
from tqdm import trange, tqdm
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision

from model import Generator, Discriminator
from utils import save_models, G_train
from utils_MD import D_train_MD, G_train_MD, compute_Lambda, LambdaNetwork
from simplex_generator import simplex_params




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()

    PARAMS = {'batch_size': args.batch_size,
          'gamma':0.5,
          'beta1':0.5,
          'beta2':0.999,
          'n_lr_steps':10,
          'lambda_training_iterations':4000,
          'epochs': args.epochs,
          'eta_lambda': 0.01,
          'lr_d': 1e-6,
          'lr_g': 1e-6,
          'epsilon': 1e-8,  # for avoiding numerical instabilities
          'samp_num_gen': 2500,
          'simplex_dim' : 9}


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    verifs_path = "verifs/MD_GAN"
    os.makedirs(verifs_path, exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim, PARAMS['simplex_dim'])).cuda()
    G_criterion = nn.BCELoss()


    # model = DataParallel(model).cuda()
    # print('Model loaded.')



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Find working device
    simplex = simplex_params(PARAMS['simplex_dim'], device) # Create Simplex
    print(f'Created simplex.')

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = PARAMS['lr_g'], betas=(PARAMS['beta1'], PARAMS['beta2']))
    D_optimizer = optim.Adam(D.parameters(), lr = PARAMS['lr_d'], betas=(PARAMS['beta1'], PARAMS['beta2']))
    epoch_step_size=PARAMS['epochs']/(PARAMS['n_lr_steps']+1) # calculate learning rate decay step size
    # lr_steps = [i for i in range(10)] 
    lr_steps=[int((i+1)*epoch_step_size) for i in range(PARAMS['n_lr_steps'])] 
    lr_g = optim.lr_scheduler.MultiStepLR(G_optimizer, lr_steps, gamma=PARAMS['gamma'])
    lr_d = optim.lr_scheduler.MultiStepLR(D_optimizer, lr_steps, gamma=PARAMS['gamma'])

    print('Start Training to get Lambda value :')

    Lambda_nn = LambdaNetwork(10).to(device)
    Lambda_train_data = torch.tensor([1.0], device=device, dtype=torch.float32, requires_grad=False)
    Lambda_optimizer = optim.Adam(Lambda_nn.parameters(), lr=PARAMS['eta_lambda'])
    Lambda_nb_it = PARAMS['lambda_training_iterations']
    # Lambda = compute_Lambda(Lambda_nn, Lambda_train_data, Lambda_optimizer, Lambda_nb_it, 
    #                         PARAMS['epsilon'], PARAMS['gamma'], simplex)
    Lambda = 67.44 # Just to not run it again and again
    print(Lambda)
    print('Start Training ...')

    n_epoch = PARAMS['epochs']

    for epoch in trange(1, n_epoch + 1, leave=True):
        pbar = tqdm(train_loader, leave=False, desc=f'Epoch {epoch}/{n_epoch}')
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.view(-1, mnist_dim)
            D_loss = D_train_MD(x, G, D, D_optimizer, Lambda=Lambda, epsilon=PARAMS['epsilon'], simplex=simplex)
            G_loss = G_train_MD(x, G, D, G_optimizer, Lambda=Lambda, epsilon=PARAMS['epsilon'], simplex=simplex)            
            # G_loss = G_train(x, G, D, G_optimizer, G_criterion)
            pbar.set_postfix(D_loss=D_loss, G_loss=G_loss)    
        
        lr_g.step(epoch) # update Generator learning rate
        lr_d.step(epoch) # update Discriminator learning rate
            
        # To see changes
        if epoch % 3 == 0:
            # save_models(G, D, 'checkpoints', args.ckpt_suffix)
            n_samples = 0
            z = torch.randn(4, 100).cuda() # Vecteur de bruit aléatoire de taille (100) pour une image. Ici batch_size images
            x = G(z)
            x = x.reshape(4, 28, 28)
            for k in range(x.shape[0]): # Tous les éléments du batch
                torchvision.utils.save_image(x[k:k+1], os.path.join(verifs_path, f'epoch{epoch}_{k}.png'))         

        if epoch % 25 == 0:
            save_models(G, D, 'checkpoints', epoch)
                    
    print('Training done')


        
