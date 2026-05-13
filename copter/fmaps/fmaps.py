import numpy as np
from scipy.interpolate import RegularGridInterpolator

def import_fmaps(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    kmax_values = list(map(int, header[1:]))
    
    kmin_values = []
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        kmin = int(parts[0])
        values = list(map(float, parts[1:]))
        kmin_values.append(kmin)
        data.append(values)
    
    data = np.array(data)

    data_mean = data.mean()
    data_std = data.std()
    data = 1 - (data - data_mean) / (np.sqrt(3) * data_std)

    kmin_values = np.array(kmin_values)
    kmax_values = np.array(kmax_values)
    
    kmin_min, kmin_max = kmin_values.min(), kmin_values.max()
    kmax_min, kmax_max = kmax_values.min(), kmax_values.max()
    
    x_norm = (kmin_values - kmin_min) / (kmin_max - kmin_min)
    y_norm = (kmax_values - kmax_min) / (kmax_max - kmax_min)
    

    interpolator = RegularGridInterpolator(
        (x_norm, y_norm),
        data,
        method='cubic',
        bounds_error=False,
        fill_value=None
    )
    

    def fitted_function(x, y):
        return interpolator(np.array([[x, y]]))[0]
    
    return fitted_function

# 示例用法
if __name__ == "__main__":

    import time
    print(time.time())
    f = copter_import('default.fmap')
    print(time.time())