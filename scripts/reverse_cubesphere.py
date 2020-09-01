
'''

File to reverse cubephere to map data, you can get input npy file by running predict.py

'''


import xarray as xr
import numpy as np
import os
import sys
sys.path.append('..')
from DLWP_CS.DLWP.remap import CubeSphereRemap

map_files = ('/home/liuyi/caiyun-weather-models/cube_remap/map_LL32x64_CS48.nc', '/home/liuyi/caiyun-weather-models/cube_remap/map_CS48_LL32x64.nc')

years = [2017, 2018]
meta_ds = []
for year in years:
    tem_data = xr.load_dataset(f'/mnt/data02/mengqy/weather_bench/cubesphere/temperature_850/tem850_cube_{year}.nc').drop('level')
    if len(meta_ds):                  # if len(dio)!=0
        meta_ds = xr.concat([meta_ds, tem_data], dim='time')
    else:                         # if len(dio)==0
        meta_ds = tem_data

time = meta_ds['time'][78::3]
dims_order = ['face', 'height', 'width']

forecast = np.load('predict.npy')
forecast = xr.DataArray(
        forecast,
        coords = [time] + [meta_ds[d] for d in dims_order],
        dims = ['time'] + [d for d in dims_order],
        name = 'forecast'
    )

forecast.to_netcdf(forecast_file + '.nc')

csr = CubeSphereRemap(to_netcdf4=True)
csr.assign_maps(*map_files)
csr.convert_from_faces(forecast_file + '.nc', forecast_file + '.tmp')
csr.inverse_remap(forecast_file + '.tmp', forecast_file, '--var', 'forecast')
os.remove(forecast_file + '.tmp')