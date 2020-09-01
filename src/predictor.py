from cubesphere_unet import CubeSphereUNet2D

import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import arrow
import numpy as np
import cv2

model_path = Path(f'./tt-20200811_152432/')
log_file = model_path / Path('predict.log')
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(configs.__dict__)

from dataset import dataset_test, dio_mean, dio_std

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

target_ratio = torch.from_numpy(np.squeeze(dio_std[0,0]))
target_constant = torch.from_numpy(np.squeeze(dio_mean[0,0]))

class Model:
    def __init__(self, configs):
        self.configs = configs
        self.network = CubeSphereUNet2D(configs, padding=1)
        if torch.cuda.is_available():
            self.network = self.network.cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        

    def RMSE(self, y_pred, y_true):
        loss = torch.mean((y_pred - y_true) ** 2, dim=[1, 2, 3, 4])
        loss = torch.mean(torch.sqrt(loss))
        return loss

    def test(self, dataloader_test):
        self.network.eval()
        loss_test = 0
        with torch.no_grad():
            for j, (test_x, test_y) in enumerate(dataloader_test):
                output, target = self.predict(test_x, test_y)
                loss_test += self.RMSE(output[:,(configs.output_length-1):configs.output_length,:,:,:], target) * target.size(0)

        loss_test = loss_test / dataset_test.__len__() * target_ratio
        return loss_test.item()

    def predict(self, test_x, test_y):
        self.network.eval()
        with torch.no_grad():
            test_x = test_x.float()
            test_y = test_y.float()
            if torch.cuda.is_available():
                test_x = test_x.cuda()
                test_y = test_y.cuda()
            if self.configs.add_now_solar and self.configs.add_future_solar:
                input_x = torch.cat((test_x, test_y[:,-12:-10,:,:,:]), dim=1)
                target = test_y[:,11:12,:,:,:]                                 # target is only temperature
                for i in range(5):
                    output = self.network(input_x)
                    if i<4:
                        input_x = torch.cat((output, test_y[:,(2*i-12):(2*i-8),:,:,:]), dim=1)
                    else:
                        input_x = torch.cat((output, test_y[:,(2*i-12):,:,:,:]), dim=1)
            elif self.configs.add_now_solar:
                input_x = test_x
                target = test_y[:,11:12,:,:,:]                                 # target is only temperature
                for i in range(5):
                    output = self.network(input_x)
                    input_x = torch.cat((output, test_y[:,(2*i-12):(2*i-10),:,:,:]), dim=1)
            elif self.configs.add_future_solar:
                input_x = torch.cat((test_x[:,:-2,:,:,:], test_y[:,-12:-10,:,:,:]), dim=1)
                target = test_y[:,11:12,:,:,:]                                 # target is only temperature
                for i in range(5):
                    output = self.network(input_x)
                    if i<4:
                        input_x = torch.cat((output, test_y[:,(2*i-10):(2*i-8),:,:,:]), dim=1)
                    else:
                        input_x = torch.cat((output, test_y[:,(2*i-10):,:,:,:]), dim=1)
            else:
                target = test_y[:,-1:,:,:,:]
                for i in range(5):
                    output = self.network(input_x)
                    input_x = output
            output = self.network(input_x)
        return output, target
    
    def load_model(self, chk_path):
        checkpoint = torch.load(chk_path)
        self.network.load_state_dict(checkpoint['net_test'])
        self.optimizer.load_state_dict(checkpoint['optimizer_tesy'])

if __name__ == '__main__':

    model = Model(configs)
    logger.info('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size, shuffle=False, num_workers=configs.n_cpu)

    model.load_model(model_path / f'checkpoint.chk')
    loss_test = model.test(dataloader_test)
    print(loss_test)
    logger.info("test loss: {0}".format(round(loss_test, 5)))
    
    #--------------plot_figure-----------------------------
    for j, (test_x, test_y) in enumerate(dataloader_test):
        predict0, target0 = model.predict(test_x, test_y)
        predict0 = predict0[:,configs.variable_num:(configs.variable_num+1),:,:,:].detach().cpu().numpy()
        if j==0:
            predict = np.array(predict0)
        else:
            predict = np.append(predict, predict0, axis=0)

    predict = predict*target_ratio + target_constant
    print('predict:', predict.shape)
    np.save("predict.npy",predict)
    #     if j ==1:
    #         predict0, target0 = model.predict(test_x, test_y)
    #         predict0 = predict0.detach().cpu().numpy()
    #         target0 = target0.detach().cpu().numpy()
    #         break
    # target0=target0*255
    # predict0=predict0*255
    # print(np.shape(target0))
    # print(np.shape(predict0))
    
    # cv2.imwrite('target0.png', target0[1,0,0])
    # cv2.imwrite('predict0.png', predict0[1,0,0])
    # cv2.imwrite('target1.png', target0[1,0,1])
    # cv2.imwrite('predict1.png', predict0[1,0,1])
    # cv2.imwrite('target2.png', target0[1,0,2])
    # cv2.imwrite('predict2.png', predict0[1,0,2])
    # cv2.imwrite('target3.png', target0[1,0,3])
    # cv2.imwrite('predict3.png', predict0[1,0,3])
    # cv2.imwrite('target4.png', target0[1,0,4])
    # cv2.imwrite('predict4.png', predict0[1,0,4])
    # cv2.imwrite('target5.png', target0[1,0,5])
    # cv2.imwrite('predict5.png', predict0[1,0,5])









