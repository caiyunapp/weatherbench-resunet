
'''

File to figure weighted loss for map data, you should run reverse_cubesphere.py first

'''


import numpy as np
import xarray as xr
from config import configs
import cv2

def weighted_RMSE(lat, y_pred, y_true):
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    weights_lat = weights_lat.reshape(1, len(weights_lat), 1)
    loss = np.mean((y_pred - y_true) ** 2 * weights_lat, axis=(1, 2))
    loss = np.mean(np.sqrt(loss))
    return loss

input_gap = configs.input_gap
input_length = configs.input_length
input_span = input_gap * (input_length - 1) + 1

years = [2017, 2018]
target = []
for year in years:
    tem_data = xr.load_dataset(f'/mnt/data02/mengqy/weather_bench/temperature_850/temperature_850hPa_{year}_5.625deg.nc')
    temp = np.array(tem_data.t)
    if len(target):
        target = np.append(target, temp, axis=0)
    else:
        target = temp

#### get index of target data
target_id = range(78, target.shape[0], 3)
target = target[target_id]
print(target.shape)

predict_data = xr.load_dataset(f'forecast.nc')
predict = np.array(predict_data.forecast)
print(predict.shape)
lat = np.array(predict_data.lat)
loss = weighted_RMSE(lat, predict, target)
print('loss:', loss)
 
# cv2.imwrite('target.png', target0[1,0,0])
# cv2.imwrite('predict.png', predict0[1,0,0])
