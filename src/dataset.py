from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import logging
from config import configs
import gc

years_train = [
    # 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
    # 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    # 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
    # 2010, 2011, 2012, 2013, 2014, 2015,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
]  # should be in sorted order
years_eval = [2016]
years_test = [2017, 2018]

################ prior information
input_gap = configs.input_gap
input_length = configs.input_length
input_span = input_gap * (input_length - 1) + 1

pred_shift = 72  # 72 or 120
pred_length = configs.output_length
pred_gap = pred_shift // pred_length
#################

logger = logging.getLogger()


solar_mean, solar_std = 0.0, 0.0
dio_mean, dio_std = 0.0, 0.0


class DatasetTrain(Dataset):
    def __init__(self, years):
        dio = []
        dio_solar = []
        for year in years:
            logger.info('loading dataset in {0}'.format(year))

            tem_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/temperature_850/tem850_cube_{year}.nc'
            )
            temp = np.array(tem_data.t)
            temp = temp[::3]

            t2m_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/T_2m/T_2m_cube_{year}.nc')
            t2m = np.array(t2m_data.t2m)
            t2m = t2m[::3]

            z500_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/Z_500/z_500_cube_{year}.nc')
            z500 = np.array(z500_data.z)
            z500 = z500[::3]

            z1000_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/Z_1000/z_1000_cube_{year}.nc')
            z1000 = np.array(z1000_data.z)
            z1000 = z1000[::3]

            thick_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/thickness_300_700/thickness_300_700_cube_{year}.nc'
            )
            thick = np.array(thick_data.thick)
            thick = thick[::3]

            if len(dio):  # if len(dio)!=0
                dio = np.append(dio,
                                np.stack((temp, t2m, z500, z1000, thick),
                                         axis=1),
                                axis=0)
            else:  # if len(dio)==0
                dio = np.stack((temp, t2m, z500, z1000, thick), axis=1)
            del temp, tem_data, t2m, t2m_data, z500, z500_data, z1000, z1000_data, thick, thick_data
            if configs.add_now_solar or configs.add_future_solar:  # if configs.add_solar==True
                solar_data = xr.load_dataset(
                    f'/data/weatherbench/CubeSphere/toa_incident_solar_radiation/solar_radiation_cube_{year}.nc'
                )
                solar = np.array(solar_data.tisr)
                solar = solar[::3]
                if len(dio_solar):
                    dio_solar = np.append(dio_solar, solar, axis=0)
                else:
                    dio_solar = solar
                del solar, solar_data
            gc.collect()

        # self.weights_lat = np.cos(np.deg2rad(temp.lat.data))
        # self.weights_lat /= self.weights_lat.mean()

        #### get index of input and target data
        input_ind = np.arange(0, input_span,
                              input_gap)  # length : input_length
        target_ind = np.arange(
            0, input_span + 2 * input_gap,
            input_gap) + input_span + input_gap - 1  # length : 2*input_length
        ind = np.concatenate([input_ind, target_ind]).reshape(
            1, 3 * input_length)  # length : 3*input_length
        max_n_sample = dio.shape[0] - (input_gap * (input_length + 3)
                                       )  # pred_length = 4*input_gap

        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones(
            (1, input_length + 2 * input_length),
            dtype=int)  # pred_length = 2*input_length
        # @ means Matrix multiplication.   [0 1 2 3 4 5 6 7 8 9 81] + [0:max,1] @ [1,1*11]

        #   input_id & target_id
        input_id = ind[:, :input_length]
        target_id = ind[:, input_length:]
        input_id = input_id.flatten()
        target_id = target_id.flatten()

        if configs.variable_num == 1:
            _, F, H, W = dio.shape
        else:
            _, C, F, H, W = dio.shape

        global dio_mean, dio_std
        dio_mean = np.mean(dio, axis=(0, 2, 3, 4), keepdims=True)
        dio_std = np.std(dio, axis=(0, 2, 3, 4), ddof=1, keepdims=True)
        dio = (dio - dio_mean) / dio_std

        self.di = dio[input_id].reshape(-1,
                                        input_length * configs.variable_num, F,
                                        H, W)
        self.do = dio[target_id].reshape(
            -1, 2 * input_length * configs.variable_num, F, H, W)

        # change the order of the channals to [temp1,temp2, t2m1, t2m2, z5001,z5002....]
        order = []
        for i in range(configs.variable_num):
            order.extend([i, i + configs.variable_num
                          ])  # order = [0,5,1,6,2,7,3,8,4,9]
        self.di = self.di[:, order]
        for i in range(2 * configs.variable_num, 3 * configs.variable_num):
            order.extend([
                i, i + configs.variable_num
            ])  # order = [0,5,1,6,2,7,3,8,4,9,10,15,11,16,12,17,13,18,14,19]
        self.do = self.do[:, order]

        del dio
        gc.collect()

        if configs.add_now_solar or configs.add_future_solar:
            global solar_mean, solar_std
            solar_mean = np.mean(dio_solar)
            solar_std = np.std(dio_solar, ddof=1)
            dio_solar = (dio_solar - solar_mean) / solar_std

            di_solar = dio_solar[input_id].reshape(-1, input_length, F, H, W)
            do_solar = dio_solar[target_id].reshape(-1, 2 * input_length, F, H,
                                                    W)
            del dio_solar
            gc.collect()
            self.do = np.append(self.do, do_solar, axis=1)
            del do_solar
            gc.collect()
            self.di = np.append(self.di, di_solar, axis=1)
            del di_solar
            gc.collect()

        assert (len(self.di.shape) == 5) and (len(self.do.shape) == 5)
        assert self.di.shape[0] == self.do.shape[0]
        logger.info(
            'loading finished, input of shape {0}, output of shape {1}'.format(
                self.di.shape, self.do.shape))

    def __len__(self):
        return self.di.shape[0]

    def __getitem__(self, index):
        return self.di[index], self.do[index]


class DatasetEval(Dataset):
    def __init__(self, years):
        dio = []
        dio_solar = []
        for year in years:
            logger.info('loading dataset in {0}'.format(year))

            tem_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/temperature_850/tem850_cube_{year}.nc'
            )
            temp = np.array(tem_data.t)
            temp = temp[::3]

            t2m_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/T_2m/T_2m_cube_{year}.nc')
            t2m = np.array(t2m_data.t2m)
            t2m = t2m[::3]

            z500_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/Z_500/z_500_cube_{year}.nc')
            z500 = np.array(z500_data.z)
            z500 = z500[::3]

            z1000_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/Z_1000/z_1000_cube_{year}.nc')
            z1000 = np.array(z1000_data.z)
            z1000 = z1000[::3]

            thick_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/thickness_300_700/thickness_300_700_cube_{year}.nc'
            )
            thick = np.array(thick_data.thick)
            thick = thick[::3]

            if len(dio):  # if len(dio)!=0
                dio = np.append(dio,
                                np.stack((temp, t2m, z500, z1000, thick),
                                         axis=1),
                                axis=0)
            else:  # if len(dio)==0
                dio = np.stack((temp, t2m, z500, z1000, thick), axis=1)
            del temp, tem_data, t2m, t2m_data, z500, z500_data, z1000, z1000_data, thick, thick_data
            if configs.add_now_solar or configs.add_future_solar:  # if configs.add_solar==True
                solar_data = xr.load_dataset(
                    f'/data/weatherbench/CubeSphere/toa_incident_solar_radiation/solar_radiation_cube_{year}.nc'
                )
                solar = np.array(solar_data.tisr)
                solar = solar[::3]
                if len(dio_solar):
                    dio_solar = np.append(dio_solar, solar, axis=0)
                else:
                    dio_solar = solar
                del solar, solar_data
            gc.collect()

        # get index of input and target data
        input_ind = np.arange(0, input_span,
                              input_gap)  # length : input_length
        target_ind = np.arange(
            0, input_span + 2 * input_gap,
            input_gap) + input_span + input_gap - 1  # length : 2*input_length
        ind = np.concatenate([input_ind, target_ind]).reshape(
            1, 3 * input_length)  # length : 3*input_length
        max_n_sample = dio.shape[0] - (input_gap * (input_length + 3)
                                       )  # pred_length = 4*input_gap

        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones(
            (1, input_length + 2 * input_length),
            dtype=int)  # pred_length = 2*input_length
        # @ means Matrix multiplication.   [0 1 2 3 4 5 6 7 8 9 81] + [0:max,1] @ [1,1*11]

        #   input_id & target_id
        input_id = ind[:, :input_length]
        target_id = ind[:, input_length:]
        input_id = input_id.flatten()
        target_id = target_id.flatten()

        if configs.variable_num == 1:
            _, F, H, W = dio.shape
        else:
            _, C, F, H, W = dio.shape

        dio = (dio - dio_mean) / dio_std

        self.di = dio[input_id].reshape(-1,
                                        input_length * configs.variable_num, F,
                                        H, W)
        self.do = dio[target_id].reshape(
            -1, 2 * input_length * configs.variable_num, F, H, W)
        del dio
        gc.collect()

        # change the order of the channals to [temp1,temp2, t2m1, t2m2, z5001,z5002....]
        order = []
        for i in range(configs.variable_num):
            order.extend([i, i + configs.variable_num
                          ])  # order = [0,5,1,6,2,7,3,8,4,9]
        self.di = self.di[:, order]
        for i in range(2 * configs.variable_num, 3 * configs.variable_num):
            order.extend([
                i, i + configs.variable_num
            ])  # order = [0,5,1,6,2,7,3,8,4,9,10,15,11,16,12,17,13,18,14,19]
        self.do = self.do[:, order]

        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = (dio_solar - solar_mean) / solar_std
            di_solar = dio_solar[input_id].reshape(-1, input_length, F, H, W)
            do_solar = dio_solar[target_id].reshape(-1, 2 * input_length, F, H,
                                                    W)
            del dio_solar
            gc.collect()
            self.do = np.append(self.do, do_solar, axis=1)
            del do_solar
            gc.collect()
            self.di = np.append(self.di, di_solar, axis=1)
            del di_solar
            gc.collect()

        assert (len(self.di.shape) == 5) and (len(self.do.shape) == 5)
        assert self.di.shape[0] == self.do.shape[0]
        logger.info(
            'loading finished, input of shape {0}, output of shape {1}'.format(
                self.di.shape, self.do.shape))

    def __len__(self):
        return self.di.shape[0]

    def __getitem__(self, index):
        return self.di[index], self.do[index]


class DatasetTest(Dataset):
    def __init__(self, years):
        dio = []
        dio_solar = []
        target = []
        for year in years:
            logger.info('loading dataset in {0}'.format(year))

            tem_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/temperature_850/tem850_cube_{year}.nc'
            )
            temp = np.array(tem_data.t)
            temp = temp[::3]

            t2m_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/T_2m/T_2m_cube_{year}.nc')
            t2m = np.array(t2m_data.t2m)
            t2m = t2m[::3]

            z500_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/Z_500/z_500_cube_{year}.nc')
            z500 = np.array(z500_data.z)
            z500 = z500[::3]

            z1000_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/Z_1000/z_1000_cube_{year}.nc')
            z1000 = np.array(z1000_data.z)
            z1000 = z1000[::3]

            thick_data = xr.load_dataset(
                f'/data/weatherbench/CubeSphere/thickness_300_700/thickness_300_700_cube_{year}.nc'
            )
            thick = np.array(thick_data.thick)
            thick = thick[::3]

            if len(dio):  # if len(dio)!=0
                dio = np.append(dio,
                                np.stack((temp, t2m, z500, z1000, thick),
                                         axis=1),
                                axis=0)
            else:  # if len(dio)==0
                dio = np.stack((temp, t2m, z500, z1000, thick), axis=1)
            if len(target):
                target = np.append(target, temp, axis=0)
            else:
                target = temp
            del temp, tem_data, t2m, t2m_data, z500, z500_data, z1000, z1000_data, thick, thick_data
            if configs.add_now_solar or configs.add_future_solar:  # if configs.add_solar==True
                solar_data = xr.load_dataset(
                    f'/data/weatherbench/CubeSphere/toa_incident_solar_radiation/solar_radiation_cube_{year}.nc'
                )
                solar = np.array(solar_data.tisr)
                solar = solar[::3]
                if len(dio_solar):
                    dio_solar = np.append(dio_solar, solar, axis=0)
                else:
                    dio_solar = solar
                del solar, solar_data
            gc.collect()

        # get index of input and target data
        input_ind = np.arange(0, input_span,
                              input_gap)  # length : input_length
        target_ind = np.arange(
            0, input_span + 10 * input_gap,
            input_gap) + input_span + input_gap - 1  # length : 2*input_length
        ind = np.concatenate([input_ind, target_ind]).reshape(
            1, 7 * input_length)  # length : 3*input_length
        max_n_sample = dio.shape[0] - input_gap * (
            input_length + 11)  # pred_length = 4*input_gap

        ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones(
            (1, input_length + 6 * input_length),
            dtype=int)  # pred_length = 2*input_length
        # @ means Matrix multiplication.   [0 1 2 3 4 5 6 7 8 9 81] + [0:max,1] @ [1,1*11]

        input_id = ind[:, :input_length]
        target_id1 = ind[:, -1:]
        target_id2 = ind[:, input_length:]
        input_id = input_id.flatten()
        target_id1 = target_id1.flatten()
        target_id2 = target_id2.flatten()

        ####
        if configs.variable_num == 1:
            _, F, H, W = dio.shape
        else:
            _, C, F, H, W = dio.shape

        # for i in range(configs.variable_num):
        #     dio[:,i] = norm_z_score(dio[:,i])
        # print('std of test t850:     ', np.std(target, ddof=1))
        # print('mean of test t850:     ', np.mean(target))

        # print('mean of variable:     ', dio_mean)
        # print('std of variable:     ', dio_std)

        dio = (dio - dio_mean) / dio_std
        target = (target - np.squeeze(dio_mean[0, 0])) / np.squeeze(dio_std[0,
                                                                            0])
        self.di = dio[input_id].reshape(-1,
                                        input_length * configs.variable_num, F,
                                        H, W)
        self.do = target[target_id1].reshape(-1, 1, F, H, W)

        # change the order of the channals to [temp1,temp2, t2m1, t2m2, z5001,z5002....]
        order = []
        for i in range(configs.variable_num):
            order.extend([i, i + configs.variable_num
                          ])  # order = [0,5,1,6,2,7,3,8,4,9]
        self.di = self.di[:, order]

        del dio, target
        gc.collect()

        if configs.add_now_solar or configs.add_future_solar:
            dio_solar = (dio_solar - solar_mean) / solar_std
            di_solar = dio_solar[input_id].reshape(-1, input_length, F, H, W)
            do_solar = dio_solar[target_id2].reshape(-1, 6 * input_length, F,
                                                     H, W)
            del dio_solar
            gc.collect()
            self.do = np.append(self.do, do_solar, axis=1)
            del do_solar
            gc.collect()
            self.di = np.append(self.di, di_solar, axis=1)
            del di_solar
            gc.collect()

        assert (len(self.di.shape) == 5) and (len(self.do.shape) == 5)
        assert self.di.shape[0] == self.do.shape[0]
        logger.info(
            'loading finished, input of shape {0}, output of shape {1}'.format(
                self.di.shape, self.do.shape))

    def __len__(self):
        return self.di.shape[0]

    def __getitem__(self, index):
        return self.di[index], self.do[index]


# train
dataset_train = DatasetTrain(years_train)
# eval and test
dataset_eval = DatasetEval(years_eval)
dataset_test = DatasetTest(years_test)
