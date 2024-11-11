import torch 
import torchvision
import os
import argparse


from model import Generator, Discriminator, Discriminator_WGAN_GP
from utils_WGP import load_model2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--G_model_name", type = str, 
                        default="G.pth")
    parser.add_argument("--D_model_name", type = str, 
                        default="D.pth")
    parser.add_argument("--checkpoints_folder", type = str,
                        default="checkpoints_wgan")
    parser.add_argument("--rejection_sampling", type = bool,
                        default="True")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model2(model, args.checkpoints_folder, args.G_model_name)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')
    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    if args.rejection_sampling : 
        discriminator = Discriminator_WGAN_GP(mnist_dim)
        discriminator = load_model2(discriminator, args.checkpoints_folder, args.D_model_name)
        discriminator = torch.nn.DataParallel(discriminator).cuda()
        discriminator.eval()

    n_samples = 0
    total_attempts = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).cuda() 
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            if args.rejection_sampling : 
                discriminator_scores = discriminator(x.view(x.size(0), -1))
                for k in range(x.shape[0]):
                    total_attempts += 1
                    acceptance_probability = discriminator_scores[k].item()
                    if n_samples < 10000 and torch.rand(1).item() < acceptance_probability:
                        torchvision.utils.save_image(x[k:k + 1], os.path.join('samples', f'{n_samples}.jpg'))
                        n_samples += 1
            else : 
                for k in range(x.shape[0]):
                    total_attempts +=1
                    if n_samples<10000:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.jpg'))         
                        n_samples += 1
    print(f"Generated {n_samples} samples after {total_attempts} attempts with rejection sampling.")

    
