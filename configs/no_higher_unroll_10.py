
z_dim = 256
g_inp = z_dim
g_hid = 128
g_out = 2

d_inp = g_out
d_hid = 128
d_out = 1

minibatch_size = 512

unrolled_steps = 10
d_learning_rate = 1e-4
g_learning_rate = 1e-3
optim_betas = (0.5, 0.999)
num_iterations = 2000
log_interval = 300
d_steps = 1
g_steps = 1

seed = 123456
use_higher = False
prefix = 'no_higher'


# z_dim = 100 # 256 de base
# g_inp = z_dim
# g_hid = 1024
# g_out = 2

# d_inp = g_out
# d_hid = 1024
# d_out = 1

# minibatch_size = 512

# unrolled_steps = 0
# d_learning_rate = 0.0002
# g_learning_rate = 0.0002
# optim_betas = (0.5, 0.999)
# num_iterations = 2000
# log_interval = 250
# d_steps = 1
# g_steps = 1

# seed = 123456
# use_higher = False
# prefix = 'no_higher'