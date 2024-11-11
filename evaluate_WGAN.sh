python generate_WGAN.py --checkpoints_folder checkpoints2_wgan \
                        --G_model_name G_wgan_pretrained150_e30.pth\
                        --D_model_name D_wgan_pretrained150_e30.pth\
                        --rejection_sampling True
python convert.py # Converts the .ubyte file in a folder of .jpg images
python evaluate.py