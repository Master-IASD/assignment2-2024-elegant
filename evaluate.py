from pytorch_fid import fid_score
import argparse
import torch

def compute_fid(real_images_path, generated_images_path, batch_size, device, dims=2048) :
    fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], 
                                                    batch_size=batch_size, 
                                                    device=device, 
                                                    dims=2048)
    return fid_value

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--real_images_path", type=str, default="test_images/")
    parser.add_argument("--generated_images_path", type=str, default="samples/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # real_images_path = "test_images/"
    # generated_images_path = "samples/"
    real_images_path = args.real_images_path
    generated_images_path = args.generated_images_path

    # fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], 
    #                                                 batch_size=args.batch_size, 
    #                                                 device=device, 
    #                                                 dims=2048)
    fid_value = compute_fid(real_images_path, generated_images_path, args.batch_size, device)

    print(f'FID: {fid_value}')