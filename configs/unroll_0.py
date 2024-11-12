
# z_dim = 256
# g_inp = z_dim
# g_hid = 128
# g_out = 2

z_dim = 100 # 256 de base
g_inp = z_dim
g_hid = 512
g_out = 2

d_inp = g_out # original
d_hid = 128
d_out = 1

# d_inp = g_out
# d_hid = 512
# d_out = 1

minibatch_size = 64

unrolled_steps = 0
d_learning_rate = 3e-5
g_learning_rate = 1e-3
optim_betas = (0.5, 0.999)
num_iterations = 3000
log_interval = 250
d_steps = 1
g_steps = 2

seed = 123456
use_higher = False
prefix = 'no_roll'