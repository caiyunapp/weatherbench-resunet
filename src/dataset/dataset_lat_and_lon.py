import numpy as np
import xarray as xr
import logging
import gc
import sys

from torch.utils.data import Dataset

sys.path.append('../')
from config import configs

years_train = [
    # 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
    # 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    # 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
    # 2010, 2011, 2012, 2013, 2014,
    #2012, 2013, 2014,
    #2003, 2004, 2005, 2006,
    2014
] # should be in sorted order
years_eval = [2015, 2016]
years_test = [2017, 2018] # 2018

################ prior information
input_gap = configs.input_gap
input_length = configs.input_length
input_span = input_gap * (input_length - 1) + 1

pred_shift = 72  # 72 or 120
pred_length = configs.output_length 
pred_gap = pred_shift // pred_length

num_per_group = (input_length + pred_length) * input_gap

logger = logging.getLogger()

solar_mean, solar_std = 0.0, 0.0
dio_mean, dio_std = 0.0, 0.0

class DatasetTrain(Dataset):
    def __init__(self, root, resolution, years):
        dio = []
        dio_solar = []
        for year in years:
            logger.info('loading dataset in {0}'.format(year))
            temp_path = '%s/%s/%s_%d_%s.nc' % (root, 'temperature_850hPa', 'temperature_850hPa', year, resolution)
            temp_2m_path = '%s/%s/%s_%d_%s.nc' % (root, '2m_temperature', '2m_temperature', year, resolution)
            geop_path = '%s/%s/%s_%d_%s.nc' % (root, 'geopotential', 'geopotential', year, resolution)

            temp = xr.open_dataset(temp_path).transpose('time', 'lat', 'lon')
            temp_2m = xr.open_dataset(temp_2m_path).transpose('time', 'lat', 'lon')
            geop = xr.open_dataset(geop_path).transpose('time', 'level', 'lat', 'lon')
            
            di1 = temp.t.data[:, np.newaxis]
            di1 = di1[::3]
            di2 = geop.z.sel(level=[300]).data - geop.z.sel(level=[700]).data
            di2 = di2[::3]
            di3 = geop.z.sel(level=[500, 1000]).data
            di3 = di3[::3]
            di4 = temp_2m.t2m.data[:, np.newaxis]
            di4 = di4[::3]
            # stack features
            dio.append(np.concatenate([di1, di2, di3, di4], axis=1))

            if configs.add_now_solar or configs.add_future_solar:           # if configs.add_solar==True
                solar_path = '%s/%s/%s_%d_%s.nc' % (root, 'toa_incident_solar_radiation', 'toa_incident_solar_radiation', year, resolution)
                solar = xr.open_dataset(solar_path).transpose('time', 'lat', 'lon')
                di5 = solar.tisr.data[:, np.newaxis]
                di5 = di5[::3]
                dio_solar.append(di5)

        self.weights_lat = np.cos(np.deg2rad(temp.lat.data))
        self.weights_lat /= self.weights_lat.mean()
        del temp, temp_2m, geop, di1, di2, di3, di4, di5
        gc.collect()
        dio = np.concatenate(dio, axis=0)
        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = np.concatenate(dio_solar, axis=0)

        input_ind = np.arange(0, input_span, input_gap)          # length : input_length
        target_ind = np.arange(0, input_span + 2*input_gap, input_gap) + input_span + input_gap - 1   # length : 2*input_length
        ind = np.concatenate([input_ind, target_ind]).reshape(1, 3*input_length)    # length : 3*input_length
        max_n_sample = dio.shape[0] - (input_gap*(input_length + 3))    # pred_length = 4*input_gap

        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + 2*input_length), dtype=int)   # pred_length = 2*input_length
        # @ means Matrix multiplication.   [0 1 2 3 4 5 6 7 8 9 81] + [0:max,1] @ [1,1*11]

        #   input_id & target_id
        input_id = ind[:,:input_length]
        target_id = ind[:,input_length:]
        input_id = input_id.flatten()
        target_id = target_id.flatten()

        if configs.variable_num==1:
            _, H, W = dio.shape
        else:
            _, C, H, W = dio.shape

        global dio_mean, dio_std
        dio_mean = np.mean(dio, axis=(0,2,3), keepdims=True)
        dio_std = np.std(dio, axis=(0,2,3), ddof=1, keepdims=True)
        dio = (dio - dio_mean) / dio_std

        self.di = dio[input_id].reshape(-1, input_length * configs.variable_num, H, W)
        self.do = dio[target_id].reshape(-1, 2*input_length * configs.variable_num, H, W)

        # change the order of the channals to [temp1,temp2, t2m1, t2m2, z5001,z5002....]
        order = []
        for i in range(configs.variable_num):
            order.extend(
                [i,i+configs.variable_num])      # order = [0,5,1,6,2,7,3,8,4,9]
        self.di = self.di[:,order]
        for i in range(2*configs.variable_num, 3*configs.variable_num):
            order.extend(
                [i,i+configs.variable_num])      # order = [0,5,1,6,2,7,3,8,4,9,10,15,11,16,12,17,13,18,14,19]
        self.do = self.do[:,order]

        del dio
        gc.collect()

        if configs.add_now_solar or configs.add_future_solar:
            global solar_mean, solar_std
            solar_mean = np.mean(dio_solar)
            solar_std = np.std(dio_solar, ddof=1)
            dio_solar = (dio_solar - solar_mean) / solar_std
            # dio_solar = norm_z_score(dio_solar)
            di_solar = dio_solar[input_id].reshape(-1, input_length, H, W)
            do_solar = dio_solar[target_id].reshape(-1, 2*input_length, H, W)
            del dio_solar
            gc.collect()
            self.do = np.append(self.do,
                                do_solar,
                                axis=1)
            del do_solar
            gc.collect()
            self.di = np.append(self.di,
                                di_solar,
                                axis=1)
            del di_solar
            gc.collect()

        assert (len(self.di.shape) == 4) and (len(self.do.shape) == 4)
        assert self.di.shape[0] == self.do.shape[0]
        logger.info('loading finished, input of shape {0}, output of shape {1}'.format(self.di.shape, self.do.shape))

    def __len__(self):
        return self.di.shape[0]

    def __getitem__(self, index):
        return self.di[index], self.do[index]

class DatasetEval(Dataset):
    def __init__(self, root, resolution, years):
        dio = []
        dio_solar = []
        for year in years:
            logger.info('loading dataset in {0}'.format(year))
            temp_path = '%s/%s/%s_%d_%s.nc' % (root, 'temperature_850hPa', 'temperature_850hPa', year, resolution)
            temp_2m_path = '%s/%s/%s_%d_%s.nc' % (root, '2m_temperature', '2m_temperature', year, resolution)
            geop_path = '%s/%s/%s_%d_%s.nc' % (root, 'geopotential', 'geopotential', year, resolution)

            temp = xr.open_dataset(temp_path).transpose('time', 'lat', 'lon')
            temp_2m = xr.open_dataset(temp_2m_path).transpose('time', 'lat', 'lon')
            geop = xr.open_dataset(geop_path).transpose('time', 'level', 'lat', 'lon')
            
            di1 = temp.t.data[:, np.newaxis]
            di1 = di1[::3]
            di2 = geop.z.sel(level=[300]).data - geop.z.sel(level=[700]).data
            di2 = di2[::3]
            di3 = geop.z.sel(level=[500, 1000]).data
            di3 = di3[::3]
            di4 = temp_2m.t2m.data[:, np.newaxis]
            di4 = di4[::3]
            # stack features
            dio.append(np.concatenate([di1, di2, di3, di4], axis=1))

            if configs.add_now_solar or configs.add_future_solar:           # if configs.add_solar==True
                solar_path = '%s/%s/%s_%d_%s.nc' % (root, 'toa_incident_solar_radiation', 'toa_incident_solar_radiation', year, resolution)
                solar = xr.open_dataset(solar_path).transpose('time', 'lat', 'lon')
                di5 = solar.tisr.data[:, np.newaxis]
                di5 = di5[::3]
                dio_solar.append(di5)
        del temp, temp_2m, geop, di1, di2, di3, di4, di5
        gc.collect()
        dio = np.concatenate(dio, axis=0)
        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = np.concatenate(dio_solar, axis=0)

        input_ind = np.arange(0, input_span, input_gap)          # length : input_length
        target_ind = np.arange(0, input_span + 2*input_gap, input_gap) + input_span + input_gap - 1   # length : 2*input_length
        ind = np.concatenate([input_ind, target_ind]).reshape(1, 3*input_length)    # length : 3*input_length
        max_n_sample = dio.shape[0] - (input_gap*(input_length + 3))    # pred_length = 4*input_gap

        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + 2*input_length), dtype=int)   # pred_length = 2*input_length
        # @ means Matrix multiplication.   [0 1 2 3 4 5 6 7 8 9 81] + [0:max,1] @ [1,1*11]

        #   input_id & target_id
        input_id = ind[:,:input_length]
        target_id = ind[:,input_length:]
        input_id = input_id.flatten()
        target_id = target_id.flatten()

        if configs.variable_num==1:
            _, H, W = dio.shape
        else:
            _, C, H, W = dio.shape

        dio = (dio - dio_mean) / dio_std

        self.di = dio[input_id].reshape(-1, input_length * configs.variable_num, H, W)
        self.do = dio[target_id].reshape(-1, 2*input_length * configs.variable_num, H, W)

        # change the order of the channals to [temp1,temp2, t2m1, t2m2, z5001,z5002....]
        order = []
        for i in range(configs.variable_num):
            order.extend(
                [i,i+configs.variable_num])      # order = [0,5,1,6,2,7,3,8,4,9]
        self.di = self.di[:,order]
        for i in range(2*configs.variable_num, 3*configs.variable_num):
            order.extend(
                [i,i+configs.variable_num])      # order = [0,5,1,6,2,7,3,8,4,9,10,15,11,16,12,17,13,18,14,19]
        self.do = self.do[:,order]

        del dio
        gc.collect()

        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = (dio_solar - solar_mean) / solar_std
            # dio_solar = norm_z_score(dio_solar)
            di_solar = dio_solar[input_id].reshape(-1, input_length, H, W)
            do_solar = dio_solar[target_id].reshape(-1, 2*input_length, H, W)
            del dio_solar
            gc.collect()
            self.do = np.append(self.do,
                                do_solar,
                                axis=1)
            del do_solar
            gc.collect()
            self.di = np.append(self.di,
                                di_solar,
                                axis=1)
            del di_solar
            gc.collect()

        assert (len(self.di.shape) == 4) and (len(self.do.shape) == 4)
        assert self.di.shape[0] == self.do.shape[0]
        logger.info('loading finished, input of shape {0}, output of shape {1}'.format(self.di.shape, self.do.shape))

    def __len__(self):
        return self.di.shape[0]

    def __getitem__(self, index):
        return self.di[index], self.do[index]


class DatasetTest(Dataset):
    def __init__(self, root, resolution, years):
        dio = []
        dio_solar = []
        target = []
        for year in years:
            logger.info('loading dataset in {0}'.format(year))
            temp_path = '%s/%s/%s_%d_%s.nc' % (root, 'temperature_850hPa', 'temperature_850hPa', year, resolution)
            temp_2m_path = '%s/%s/%s_%d_%s.nc' % (root, '2m_temperature', '2m_temperature', year, resolution)
            geop_path = '%s/%s/%s_%d_%s.nc' % (root, 'geopotential', 'geopotential', year, resolution)

            temp = xr.open_dataset(temp_path).transpose('time', 'lat', 'lon')
            temp_2m = xr.open_dataset(temp_2m_path).transpose('time', 'lat', 'lon')
            geop = xr.open_dataset(geop_path).transpose('time', 'level', 'lat', 'lon')
            
            di1 = temp.t.data[:, np.newaxis]
            di1 = di1[::3]
            di2 = geop.z.sel(level=[300]).data - geop.z.sel(level=[700]).data
            di2 = di2[::3]
            di3 = geop.z.sel(level=[500, 1000]).data
            di3 = di3[::3]
            di4 = temp_2m.t2m.data[:, np.newaxis]
            di4 = di4[::3]
            # stack features
            dio.append(np.concatenate([di1, di2, di3, di4], axis=1))
            target.append(di1)

            if configs.add_now_solar or configs.add_future_solar:           # if configs.add_solar==True
                solar_path = '%s/%s/%s_%d_%s.nc' % (root, 'toa_incident_solar_radiation', 'toa_incident_solar_radiation', year, resolution)
                solar = xr.open_dataset(solar_path).transpose('time', 'lat', 'lon')
                di5 = solar.tisr.data[:, np.newaxis]
                di5 = di5[::3]
                dio_solar.append(di5)

        self.weights_lat = np.cos(np.deg2rad(temp.lat.data))
        self.weights_lat /= self.weights_lat.mean()

        del temp, temp_2m, geop, di1, di2, di3, di4, di5
        gc.collect()
        dio = np.concatenate(dio, axis=0)
        target = np.concatenate(target, axis=0)
        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = np.concatenate(dio_solar, axis=0)

        #### get index of input and target data
        input_ind = np.arange(0, input_span, input_gap)          # length : input_length
        target_ind = np.arange(0, input_span + 10*input_gap, input_gap) + input_span + input_gap - 1   # length : 2*input_length
        ind = np.concatenate([input_ind, target_ind]).reshape(1, 7*input_length)    # length : 3*input_length
        max_n_sample = dio.shape[0] - input_gap*(input_length + 11)    # pred_length = 4*input_gap

        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length + 6*input_length), dtype=int)   # pred_length = 2*input_length
        # @ means Matrix multiplication.   [0 1 2 3 4 5 6 7 8 9 81] + [0:max,1] @ [1,1*11]
        input_id = ind[:,:input_length]
        target_id1 = ind[:,-1:]
        target_id2 = ind[:,input_length:]
        input_id = input_id.flatten()
        target_id1 = target_id1.flatten()
        target_id2 = target_id2.flatten()
        #ind = ind.flatten()
        ####
        if configs.variable_num==1:
            _, H, W = dio.shape
        else:
            _, C, H, W = dio.shape
        
        dio = (dio - dio_mean) / dio_std
        target = (target - np.squeeze(dio_mean[0,0])) / np.squeeze(dio_std[0,0])
        self.di = dio[input_id].reshape(-1, input_length * configs.variable_num, H, W)
        self.do = target[target_id1].reshape(-1, 1, H, W)

        # change the order of the channals to [temp1,temp2, t2m1, t2m2, z5001,z5002....]
        order = []
        for i in range(configs.variable_num):
            order.extend(
                [i,i+configs.variable_num])      # order = [0,5,1,6,2,7,3,8,4,9]
        self.di = self.di[:,order]

        del dio, target
        gc.collect()

        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = (dio_solar - solar_mean) / solar_std
            di_solar = dio_solar[input_id].reshape(-1, input_length, H, W)
            do_solar = dio_solar[target_id2].reshape(-1, 6*input_length, H, W)
            del dio_solar
            gc.collect()
            self.do = np.append(self.do,
                                do_solar,
                                axis=1)
            del do_solar
            gc.collect()
            self.di = np.append(self.di,
                                di_solar,
                                axis=1)
            del di_solar
            gc.collect()

        assert (len(self.di.shape) == 4) and (len(self.do.shape) == 4)
        assert self.di.shape[0] == self.do.shape[0]
        logger.info('loading finished, input of shape {0}, output of shape {1}'.format(self.di.shape, self.do.shape))

    def __len__(self):
        return self.di.shape[0]

    def __getitem__(self, index):
        return self.di[index], self.do[index]
# train_xy
dataset_train = DatasetTrain('/mnt/data02/mengqy/weather_bench', '5.625deg', years_train)
# eval_x and test_y
dataset_eval = DatasetEval('/mnt/data02/mengqy/weather_bench', '5.625deg', years_eval)
dataset_test = DatasetTest('/mnt/data02/mengqy/weather_bench', '5.625deg', years_test)

