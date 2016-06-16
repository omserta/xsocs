import os
from kmap.process import peak_fit

import numpy as np

# output directory (some temporary files will also be written there)
workdir = '/path/to/workdir/'

# path to the hdf5 file written by the img_to_qspace function
qspace_f = '/path/to/qspace.h5'

# result file
result_file = os.path.join(workdir, 'results.txt')

results, success = peak_fit.peak_fit(qspace_f,
                                     indices=range(10),
                                     fit_type=peak_fit.FitTypes.LEASTSQ,
                                     n_proc=None)

with open(result_file, 'w+') as res_f:
    res_f.write('# X Y qx qy qz q I valid\n')
    for i, s in enumerate(success):
        x_pos = results[i, 0]
        y_pos = results[i, 1]
        xpeak = results[i, 3]
        ypeak = results[i, 6]
        zpeak = results[i, 9]
        xpeak_max = results[i, 2]
        q = np.sqrt(xpeak**2 + ypeak**2 + zpeak**2)
        r = (x_pos, y_pos, xpeak, ypeak, zpeak, q, xpeak_max, s)
        res_str = '{0} {1} {2} {3} {4} {5} {6} {7} ({8})\n'.format(i, *r)
        res_f.write(res_str)
