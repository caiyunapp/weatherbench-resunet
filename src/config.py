import torch.nn as nn
import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 8
configs.batch_size = 24
configs.lr = 0.001
configs.weight_decay = 0.0
configs.display_interval = 25
configs.num_epochs = 300
configs.early_stopping = True
configs.patience = 20
configs.gradient_clipping = False
configs.clipping_threshold = 3.

# data related
configs.variable_num = 5

configs.input_dim = 16  # input_length * variable_num + constants_num * add_constants + input_length * add_solar(now & future)
configs.output_dim = 10  # output_length * variable_num
configs.input_length = 2
configs.output_length = 2

configs.input_gap = 2

configs.add_constants = True
configs.add_now_solar = True
configs.add_future_solar = True
# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.add_selayer = True

configs.layers_per_block = (2, 2, 2, 2, 2)
configs.hidden_channels = (32, 64, 128, 64, 32)
assert len(configs.layers_per_block) == len(configs.hidden_channels)
