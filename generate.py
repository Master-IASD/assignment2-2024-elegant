import torch 
import torchvision
import os
import argparse


from model import Generator, Discriminator
from utils import load_model, load_discriminator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim)
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model)
    model.eval()

    discriminator = Discriminator(mnist_dim)
    discriminator = load_discriminator(discriminator, 'checkpoints')
    discriminator = torch.nn.DataParallel(discriminator)
    discriminator.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    total_attempts = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100)
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            discriminator_scores = discriminator(x.view(x.size(0), -1))
            for k in range(x.shape[0]):
                total_attempts += 1
                acceptance_probability = discriminator_scores[k].item()
                if n_samples < 10000 and torch.rand(1).item() < acceptance_probability:
                    torchvision.utils.save_image(x[k:k + 1], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1
    print(f"Generated {n_samples} samples after {total_attempts} attempts with rejection sampling.")