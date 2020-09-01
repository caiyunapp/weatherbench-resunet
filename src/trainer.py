from models.hyperbolic import HyperModel
# from models.warp import WarpModel
# from models.unet import CubeSphereUNet2D

import os
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from pathlib import Path
import numpy as np
import arrow

from dataset import dataset_train, dataset_eval, dataset_test, dio_std


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


time_str = arrow.now().format('YYYYMMDD_HHmmss')
model_path = Path(f'./tt-{time_str}')
model_path.mkdir(exist_ok=True)
log_file = model_path / Path('train.log')
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(configs.__dict__)

continue_training = False     # if continue training, load model
s=(0,-1,-1)

network = HyperModel()
if torch.cuda.is_available():
    network = network.cuda()

network = nn.DataParallel(network, output_device=0)

optimizer = torch.optim.Adam(network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True, min_lr=0.00004)
target_ratio = torch.from_numpy(np.squeeze(dio_std[0,0]))


def RMSE(y_pred, y_true):
    loss = torch.mean((y_pred - y_true) ** 2, dim=[1, 2, 3, 4])
    loss = torch.mean(torch.sqrt(loss))
    return loss


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
        input_x = torch.cat((input_x, input_y[:,-4:-2,:,:,:]), dim=1)     # temperature + now_solar + future_solar
    elif configs.add_future_solar:                   # if self.configs.add_future_solar and not self.configs.add_now_solar
        input_x[:,-2:,:,:,:] = input_y[:,-4:-2,:,:,:]                     # temperature + future_solar

    output1 = network(input_x)
    if configs.add_now_solar and configs.add_future_solar:
        input2 = torch.cat((output1, input_y[:,-4:,:,:,:]), dim=1)
        target = input_y[:,0:-4,:,:,:]                    # targets don't include solar
    elif configs.add_now_solar:
        input2 = torch.cat((output1, input_y[:,-4:-2,:,:,:]), dim=1) # concatenate temperature & solar
        target = input_y[:,0:-4,:,:,:]                    # targets don't include solar
    elif configs.add_future_solar:
        input2 = torch.cat((output1, input_y[:,-2:,:,:,:]), dim=1)
        target = input_y[:,0:-4,:,:,:]                    # targets don't include solar
    else:
        input2 = output1
        target = input_y
    
    output2 = network(input2)
    output = torch.cat((output1, output2), dim=1)
    loss = RMSE(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_all(dataloader_train, dataloader_eval, dataloader_test):
    count_eval = 0
    best_eval = 1000
    count_test = 0
    best_test = 1000
    for i in range(configs.num_epochs):
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
                input_x = torch.cat((input_x, input_y[:,-4:-2,:,:,:]), dim=1)     # temperature + now_solar + future_solar
            elif configs.add_future_solar:                   # if self.configs.add_future_solar and not self.configs.add_now_solar
                input_x[:,-2:,:,:,:] = input_y[:,-4:-2,:,:,:]                     # temperature + future_solar
            
            if torch.cuda.is_available():
                input_x = input_x.cuda()
                input_y = input_y.cuda()

            output1 = network(input_x)

            if configs.add_now_solar and configs.add_future_solar:
                input2 = torch.cat((output1, input_y[:,-4:,:,:,:]), dim=1)
                target = input_y[:,0:-4,:,:,:]                    # targets don't include solar
            elif configs.add_now_solar:
                input2 = torch.cat((output1, input_y[:,-4:-2,:,:,:]), dim=1) # concatenate temperature & solar
                target = input_y[:,0:-4,:,:,:]                    # targets don't include solar
            elif configs.add_future_solar:
                input2 = torch.cat((output1, input_y[:,-2:,:,:,:]), dim=1)
                target = input_y[:,0:-4,:,:,:]                    # targets don't include solar
            else:
                input2 = output1
                target = input_y
            
            output2 = network(input2)
            output = torch.cat((output1, output2), dim=1)

            loss_val += RMSE(output, target) * target.size(0)

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
                input_x = torch.cat((test_x, test_y[:,-12:-10,:,:,:]), dim=1)
                target = test_y[:,:1,:,:,:]                                 # target is temperature in 3 days later
                for i in range(5):
                    output = network(input_x)
                    if i<4:
                        input_x = torch.cat((output, test_y[:,(2*i-12):(2*i-8),:,:,:]), dim=1)
                    else:
                        input_x = torch.cat((output, test_y[:,(2*i-12):,:,:,:]), dim=1)
            elif configs.add_now_solar:
                input_x = test_x
                target = test_y[:,:1,:,:,:]                                 # target is temperature in 3 days later
                for i in range(5):
                    output = network(input_x)
                    input_x = torch.cat((output, test_y[:,(2*i-12):(2*i-10),:,:,:]), dim=1)
            elif configs.add_future_solar:
                input_x = torch.cat((test_x[:,:-2,:,:,:], test_y[:,-12:-10,:,:,:]), dim=1)
                target = test_y[:,:1,:,:,:]                                 # target is temperature in 3 days later
                for i in range(5):
                    output = network(input_x)
                    if i<4:
                        input_x = torch.cat((output, test_y[:,(2*i-10):(2*i-8),:,:,:]), dim=1)
                    else:
                        input_x = torch.cat((output, test_y[:,(2*i-10):,:,:,:]), dim=1)
            else:
                target = test_y[:,:1,:,:,:]
                for i in range(5):
                    output = network(input_x)
                    input_x = output

            output = network(input_x)
            loss_test += RMSE(output[:,(configs.output_length-1):configs.output_length,:,:,:], target) * target.size(0)

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
    dataloader_train = DataLoader(dataset_train, batch_size=configs.batch_size, shuffle=True, num_workers=configs.n_cpu)
    logger.info('loading eval dataloader')
    dataloader_eval = DataLoader(dataset_eval, batch_size=configs.batch_size, shuffle=False, num_workers=configs.n_cpu)
    logger.info('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size, shuffle=False, num_workers=configs.n_cpu)

    if continue_training:
        continue_train(f'./tt-20200827_113604/checkpoint.chk')
    
    train_all(dataloader_train, dataloader_eval, dataloader_test)
    logger.info('\n######training finished!########\n')

    load_model_test(model_path / f'checkpoint.chk')
    loss_test = test(dataloader_test)
    logger.info("test loss from best test model: {0}".format(round(loss_test, 5)))
