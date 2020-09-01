
'''

File to make cubephere data, input for netCDF only.

'''


import xarray as xr
import numpy as np
import netCDF4 as nc

import os
import sys
sys.path.append('..')
from DLWP_CS.DLWP.remap import CubeSphereRemap

years=[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005]
for year in years:
    data_directory = '/mnt/data02/mengqy/weather_bench/toa_incident_solar_radiation'
    output_directory = '/mnt/data24/cubesphere/toa_incident_solar_radiation'
    processed_file = f'%s/toa_incident_solar_radiation_{year}_5.625deg.nc' % data_directory
    remapped_file = f'%s/solar_radiation_cube_{year}.nc' % output_directory

    raw_data = xr.load_dataset(processed_file)
    #raw_data['z'] = raw_data.z[:,10,:,:]
    #raw_data['thick'] = raw_data.z[:,4,:,:] - raw_data.z[:,8,:,:]                    # 10 means 1000 hPa; 6 means 500 hPa; 4 means 300 hPa; 8 means 700 hPa
    if 'level' in raw_data.variables and 'time' in raw_data.variables:
        time_series = np.array(raw_data.coords['time'])
        cleaned_data = raw_data.drop('time').drop('level')
        cleaned_file = '%s/tem850_ll.nc' % output_directory
        cleaned_data.to_netcdf(cleaned_file)
    elif 'time' in raw_data.variables:
        time_series = np.array(raw_data.coords['time'])
        cleaned_data = raw_data.drop('time')
        cleaned_file = '%s/tem850_ll.nc' % output_directory
        cleaned_data.to_netcdf(cleaned_file)
    elif 'level' in raw_data.variables:
        cleaned_data = raw_data.drop('level')
        cleaned_file = '%s/tem850_ll.nc' % output_directory
        cleaned_data.to_netcdf(cleaned_file)
    else:
        cleaned_file = processed_file
    
    csr = CubeSphereRemap(path_to_remapper='/home/liuyi/.pyenv/versions/anaconda3-2020.02/pkgs/tempest-remap-2.0.3-h5f743cb_0/bin')
    csr.generate_offline_maps_from_file(cleaned_file, res=48, remove_meshes=False)
    csr.remap(cleaned_file, '%s/temp.nc' % data_directory, '--var', 'tisr')
    
    csr.convert_to_faces('%s/temp.nc' % data_directory,
            remapped_file,
            coord_file=processed_file)

    os.remove('%s/temp.nc' % data_directory)
