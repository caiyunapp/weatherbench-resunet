import os
import importlib
import argparse
import logging
import arrow
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import configs
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='0', help="index of gpu")
parser.add_argument("-c", "--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-m", "--model", type=str, default=None, help="metrological model to load")
parser.add_argument("-k", "--check", type=str, default=None, help="checkpoint file to load")
# parser.add_argument("-t", "--check", type=str, default=None, help="checkpoint file to load")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

time_str = arrow.now().format('YYYYMMDD_HHmmss')
model_path = Path(f'./tt-{time_str}')
model_path.mkdir(exist_ok=True)
log_file = model_path / Path('train.log')
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(configs.__dict__)

if opt.check:
    continue_training = True       # if continue training, load model
else:
    continue_training = False
trans_train = True           # if trans_train, run train_trans function

network_package = None
model_name = opt.model
try:
    network_package = importlib.import_module('models.%s' % model_name, package=None)
except ImportError as e:
    logger.exception(e)
    exit(1)

if model_name == 'hyperbolic_cubesphere' or model_name == 'warp_cubesphere' or model_name == 'unet':
    is_cubesphere = True
    from dataset.dataset_cubesphere import dataset_train, dataset_eval, dataset_test, dio_std
else:
    is_cubesphere = False
    from dataset.dataset_lat_and_lon import dataset_train, dataset_eval, dataset_test, dio_std

network = network_package.model
if torch.cuda.is_available():
    network = network.cuda()

network = nn.DataParallel(network)

optimizer = torch.optim.Adam(network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True, min_lr=0.00004)
target_ratio = torch.from_numpy(np.squeeze(dio_std[0,0]))

def RMSE(y_pred, y_true):
    loss = torch.mean((y_pred - y_true) ** 2, dim=[1, 2, 3, 4])
    loss = torch.mean(torch.sqrt(loss))
    return loss

def weighted_RMSE(y_pred, y_true):
    weights_lat = dataset_train.weights_lat
    weights_lat = weights_lat.reshape(1, 1, len(weights_lat), 1)
    weights_lat = torch.from_numpy(weights_lat).float().to(y_true.device)
    loss = torch.mean((y_pred - y_true) ** 2 * weights_lat, dim=[1, 2, 3])
    loss = torch.mean(torch.sqrt(loss))
    return loss

def train_trans(input_x, parts=4):
    x_size = tuple(input_x.size())
    assert (x_size[-1] % parts) == 0
    for i in range(parts):
        if i==0:
            input_x = input_x
            output_seq = network(input_x).unsqueeze(dim=0)
        else:
            input_x = torch.cat((input_x[:,:,:,(x_size[-1]//parts):], input_x[:,:,:,:(x_size[-1]//parts)]), dim=-1)
            output_seq = torch.cat((output_seq, network(input_x).unsqueeze(dim=0)), dim=0)

        input2 = torch.flip(input_x, dims=(-2,))
        output_seq = torch.cat((output_seq, torch.flip(network(input2), dims=(-2,)).unsqueeze(dim=0)), dim=0)
    
    output = torch.mean(output_seq, dim=0)
    return output

def train_once(input_x, input_y):
    network.train()
    input_x = input_x.float()
    input_y = input_y.float()
    if torch.cuda.is_available():
        input_x = input_x.cuda()
        input_y = input_y.cuda()

    optimizer.zero_grad()
    # add_solar
    if configs.add_now_solar and configs.add_future_solar:
        input_x = torch.cat((input_x, input_y[:,-4:-2]), dim=1)     # temperature + now_solar + future_solar
    elif configs.add_future_solar:                   # if self.configs.add_future_solar and not self.configs.add_now_solar
        input_x[:,-2:] = input_y[:,-4:-2]                     # temperature + future_solar

    if trans_train:
        output1 = train_trans(input_x)
    else:
        output1 = network(input_x)
    
    if configs.add_now_solar and configs.add_future_solar:
        input2 = torch.cat((output1, input_y[:,-4:]), dim=1)
        target = input_y[:,0:-4]                    # targets don't include solar
    elif configs.add_now_solar:
        input2 = torch.cat((output1, input_y[:,-4:-2]), dim=1) # concatenate temperature & solar
        target = input_y[:,0:-4]                    # targets don't include solar
    elif configs.add_future_solar:
        input2 = torch.cat((output1, input_y[:,-2:]), dim=1)
        target = input_y[:,0:-4]                    # targets don't include solar
    else:
        input2 = output1
        target = input_y
    
    if trans_train:
        output2 = train_trans(input2)
    else:
        output2 = network(input2)

    output = torch.cat((output1, output2), dim=1)

    if is_cubesphere:
        loss = RMSE(output, target)
    else: 
        loss = weighted_RMSE(output, target)
    loss.backward()
    if configs.gradient_clipping:
        nn.utils.clip_grad_norm_(network.parameters(), configs.clipping_threshold)
    optimizer.step()
    return loss.item()


def train_all(dataloader_train, dataloader_eval, dataloader_test):
    count_eval = 0
    best_eval = 1000
    count_test = 0
    best_test = 1000
    for i in range(opt.n_epochs):
        logger.info('\nepoch: {0}'.format(i+1))
        for j, (input_x, input_y) in enumerate(dataloader_train):
            loss_train = train_once(input_x, input_y)

            if j % configs.display_interval == 0:
                logger.info('batch training loss: {:.5f}'.format(loss_train))

        # evaluation
        loss_eval = valid(dataloader_eval)
        logger.info('epoch eval loss: {:.5f}'.format(loss_eval))
        lr_scheduler.step(loss_eval)
        if loss_eval >= best_eval:
            count_eval += 1
            logger.info('eval loss is not improved for {0} epoch'.format(count_eval))
        else:
            count_eval = 0
            logger.info('eval loss is improved from {:.5f} to {:.5f}'.format(best_eval, loss_eval))
            best_eval = loss_eval

        loss_test = test(dataloader_test)
        logger.info('epoch test loss: {:.5f}'.format(loss_test))
        if loss_test >= best_test:
            count_test += 1
            logger.info('test loss is not improved for {0} epoch'.format(count_test))
        else:
            count_test = 0
            logger.info('test loss is improved from {:.5f} to {:.5f}, saving model'.format(best_test, loss_test))
            save_model_test()
            best_test = loss_test

        if count_test == configs.patience:
            logger.info('early stopping reached, best test loss is {:5f}'.format(best_test))
            break


def valid(dataloader_val):
    network.eval()
    loss_val = 0
    with torch.no_grad():
        for j, (input_x, input_y) in enumerate(dataloader_val):
            input_x = input_x.float()
            input_y = input_y.float()
            # add_solar
            if configs.add_now_solar and configs.add_future_solar:
                input_x = torch.cat((input_x, input_y[:,-4:-2]), dim=1)     # temperature + now_solar + future_solar
            elif configs.add_future_solar:                   # if self.configs.add_future_solar and not self.configs.add_now_solar
                input_x[:,-2:] = input_y[:,-4:-2]                     # temperature + future_solar
            
            if torch.cuda.is_available():
                input_x = input_x.cuda()
                input_y = input_y.cuda()

            if trans_train:
                output1 = train_trans(input_x)
            else:
                output1 = network(input_x)

            if configs.add_now_solar and configs.add_future_solar:
                input2 = torch.cat((output1, input_y[:,-4:]), dim=1)
                target = input_y[:,0:-4]                    # targets don't include solar
            elif configs.add_now_solar:
                input2 = torch.cat((output1, input_y[:,-4:-2]), dim=1) # concatenate temperature & solar
                target = input_y[:,0:-4]                    # targets don't include solar
            elif configs.add_future_solar:
                input2 = torch.cat((output1, input_y[:,-2:]), dim=1)
                target = input_y[:,0:-4]                    # targets don't include solar
            else:
                input2 = output1
                target = input_y
            
            if trans_train:
                output2 = train_trans(input2)
            else:
                output2 = network(input2)
            output = torch.cat((output1, output2), dim=1)

            if is_cubesphere:
                loss_val += RMSE(output, target) * target.size(0)
            else: 
                loss_val += weighted_RMSE(output, target) * target.size(0)

    loss_val = loss_val / dataset_eval.__len__()
    return loss_val.item()
    

def test(dataloader_test):
    network.eval()
    loss_test = 0
    with torch.no_grad():
        for j, (test_x, test_y) in enumerate(dataloader_test):
            test_x = test_x.float()
            test_y = test_y.float()
            
            if torch.cuda.is_available():
                test_x = test_x.cuda()
                test_y = test_y.cuda()

            if configs.add_now_solar and configs.add_future_solar:
                input_x = torch.cat((test_x, test_y[:,-12:-10]), dim=1)
                target = test_y[:,:1]                                 # target is temperature in 3 days later
                for i in range(5):

                    if trans_train:
                        output = train_trans(input_x)
                    else:
                        output = network(input_x)

                    if i<4:
                        input_x = torch.cat((output, test_y[:,(2*i-12):(2*i-8)]), dim=1)
                    else:
                        input_x = torch.cat((output, test_y[:,(2*i-12):]), dim=1)
            elif configs.add_now_solar:
                input_x = test_x
                target = test_y[:,:1]                                 # target is temperature in 3 days later
                for i in range(5):
                    if trans_train:
                        output = train_trans(input_x)
                    else:
                        output = network(input_x)

                    input_x = torch.cat((output, test_y[:,(2*i-12):(2*i-10)]), dim=1)
            elif configs.add_future_solar:
                input_x = torch.cat((test_x[:,:-2], test_y[:,-12:-10]), dim=1)
                target = test_y[:,:1]                                 # target is temperature in 3 days later
                for i in range(5):
                    if trans_train:
                        output = train_trans(input_x)
                    else:
                        output = network(input_x)

                    if i<4:
                        input_x = torch.cat((output, test_y[:,(2*i-10):(2*i-8)]), dim=1)
                    else:
                        input_x = torch.cat((output, test_y[:,(2*i-10):]), dim=1)
            else:
                target = test_y[:,:1]
                for i in range(5):
                    if trans_train:
                        output = train_trans(input_x)
                    else:
                        output = network(input_x)

                    input_x = output

            if trans_train:
                output = train_trans(input_x)
            else:
                output = network(input_x)

            if is_cubesphere:
                loss_test += RMSE(output[:,(configs.output_length-1):configs.output_length], target) * target.size(0)
            else: 
                loss_test += weighted_RMSE(output[:,(configs.output_length-1):configs.output_length], target) * target.size(0)

    loss_test = loss_test / dataset_test.__len__()
    loss_test = loss_test * target_ratio
    return loss_test.item()
    
def save_model_test():
    torch.save({'net_test': network.state_dict(), 
                'optimizer_test':optimizer.state_dict()}, model_path / f'checkpoint.chk')

def load_model_test(chk_path):
    checkpoint = torch.load(chk_path)
    network.load_state_dict(checkpoint['net_test'])
    optimizer.load_state_dict(checkpoint['optimizer_test'])

def continue_train(chk_path):
    checkpoint = torch.load(chk_path)
    network.load_state_dict(checkpoint['net_test'])
    optimizer.load_state_dict(checkpoint['optimizer_test'])


if __name__ == '__main__':

    logger.info('loading train dataloader')
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    logger.info('loading eval dataloader')
    dataloader_eval = DataLoader(dataset_eval, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    logger.info('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    if continue_training:
        continue_train(opt.check)
    
    train_all(dataloader_train, dataloader_eval, dataloader_test)
    logger.info('\n######training finished!########\n')

    load_model_test(model_path / f'checkpoint.chk')
    loss_test = test(dataloader_test)
    logger.info("test loss from best test model: {0}".format(round(loss_test, 5)))
