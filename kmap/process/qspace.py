import os
import time

import h5py
import numpy as np
import xrayutilities as xu

from scipy.signal import medfilt
from scipy.optimize import leastsq

from silx.math import histogramnd_get_lut, histogramnd_from_lut

positioners_tpl = '/{0}/instrument/positioners'
img_data_tpl = '/{0}/measurement/image_data/data'
measurement_tpl = '/{0}/measurement'


en = 8000.
chpdeg = [318., 318.]
cch = [140, 322]
nav = [4, 4]
nx, ny, nz = 28, 154, 60
roi = [0, 516, 0, 516]

def img_2_qpeak(master_fn,
                 workdir,
                 #roi=None,
                 nav=(4, 4),
                 n_bins=(28, 154, 60)):
    """
    TODO : roi
    """
                     
    ta = time.time()

    master_fn = os.path.join(base, 'master.h5')
    tmp_fn = os.path.join(workdir, 'tmp.h5')
    res_fn = os.path.join(workdir, 'results.txt')
    
    nx, ny, nz = n_bins

    qconv = xu.experiment.QConversion(['y-'],
                                      ['z+', 'y-'],
                                      [1, 0, 0])

    # convention for coordinate system:
    # - x downstream
    # - z upwards
    # - y to the "outside"
    # (righthanded)
    hxrd = xu.HXRD([1, 0, 0],
                   [0, 0, 1],
                   en=en,
                   qconv=qconv)

    with h5py.File(master_fn, 'r') as master_h5:

        entries = sorted(master_h5.keys())
        
        n_entries = len(entries)

        n_xy_pos = None
        n_images = None

        # retrieving some info from the first image to initialize some arrays
        img_data = master_h5[img_data_tpl.format(entries[0])][0]
        
        if roi is None:
            in_roi = [0, img_data.shape[1], 0, img_data.shape[2]]
        else:
            #TODO : values check
            in_roi = roi
            img_slice = (slice(roi[0], roi[1]), slice(roi[2], roi[3]))
            
        positioners = master_h5[positioners_tpl.format(entries[0])]
        n_xy_pos = len(measurement['imgnr'])
        n_images = img_data.shape[0]
        img_x, img_y = img_data.shape[1:3]
        
        hxrd.Ang2Q.init_area('z-',
                             'y+',
                             cch1=cch[0],
                             cch2=cch[1],
                             Nch1=img_x,
                             Nch2=img_y,
                             chpdeg1=chpdeg[0],
                             chpdeg2=chpdeg[1],
                             Nav=nav,
                             roi=roi)

        # shape of the array that will store the qx/qy/qz for all rocking angles
        q_shape = (n_entries,
                   img_data.shape[1] // nav[0] * img_data.shape[2] // nav[1],
                   3)

        # then the array
        q_ar = np.zeros(q_shape, dtype=np.float64)

        for entry_idx, entry in enumerate(entries):
            positioners = master_h5[positioners_tpl.format(entry)]
            img_data = master_h5[img_data_tpl.format(entry)]
            measurement = master_h5[measurement_tpl.format(entry)]

            n_xy = len(measurement['imgnr'])
            n_img = img_data.shape[0]
            img_shape = img_data.shape

            # some minimal checks
            # TODO : are we sure the number of images will always be the same
            #   (e.g : 1 failed x,y scan that is done a second time)?
            if n_xy != n_xy_pos:
                raise ValueError('TODO')

            # some minimal checks
            if n_img != n_images:
                raise ValueError('TODO')
            if n_img != n_xy_pos:
                raise ValueError('TODO')
            if img_shape[1] != img_x:
                raise ValueError('TODO')
            if img_shape[2] != img_y:
                raise ValueError('TODO')
                
            eta = np.float64(positioners['eta'][()])
            nu = np.float64(positioners['nu'][()])
            delta = np.float64(positioners['del'][()])

            qx, qy, qz = hxrd.Ang2Q.area(eta, nu, delta)
            q_ar[entry_idx, :, 0] = qx.reshape(-1)
            q_ar[entry_idx, :, 1] = qy.reshape(-1)
            q_ar[entry_idx, :, 2] = qz.reshape(-1)

        # custom bins range to have the same histo as xrayutilities.gridder3d
        # the last bin extends beyond q_max
        qx_min = q_ar[:, :, 0].min()
        qy_min = q_ar[:, :, 1].min()
        qz_min = q_ar[:, :, 2].min()
        qx_max = q_ar[:, :, 0].max()
        qy_max = q_ar[:, :, 1].max()
        qz_max = q_ar[:, :, 2].max()
        
        step_x = (qx_max - qx_min)/(nx-1.)
        step_y = (qy_max - qy_min)/(ny-1.)
        step_z = (qz_max - qz_min)/(nz-1.)

        bins_rng_x = [qx_min - step_x/2., qx_min + (qx_max - qx_min + step_x) - step_x/2.]
        bins_rng_y = [qy_min - step_y/2., qy_min + (qy_max - qy_min + step_y) - step_y/2.]
        bins_rng_z = [qz_min - step_z/2., qz_min + (qz_max - qz_min + step_z) - step_z/2.]
        bins_rng = [bins_rng_x, bins_rng_y, bins_rng_z]

        qx_idx = qx_min + step_x * np.arange(0, nx, dtype=np.float64)
        qy_idx = qy_min + step_y * np.arange(0, ny, dtype=np.float64)
        qz_idx = qz_min + step_z * np.arange(0, nz, dtype=np.float64)
        
        img_shape_1 = img_shape[1]//nav[0], nav[0], img_shape[2]
        img_shape_2 = img_shape_1[0], img_shape_1[2]//nav[1], nav[1]
        sum_axis_1 = 1
        sum_axis_2 = 2
        # img_shape_1 = img_shape[1], img_shape[2]/nav[1], nav[1]
        # img_shape_2 = img_shape[1]//nav[0], nav[0], img_shape_1[1]
        # sum_axis_1 = 2
        # sum_axis_2 = 1
        avg_weight = 1./(nav[0]*nav[1])
        
        h_lut = None
        histo = np.zeros([nx, ny, nz], dtype=np.int32)

        for h_idx in range(n_entries):
            lut = histogramnd_get_lut(q_ar[h_idx, ...],
                                      bins_rng,
                                      [nx, ny, nz],
                                      last_bin_closed=True)
            if h_lut is None:
                h_lut = np.zeros((n_entries,) + lut[0].shape,
                                 dtype=lut[0].dtype)
                
            h_lut[h_idx, :] = lut[0]
            histo += lut[1]
            
        mask = histo>0
        
        # array to store the results
        # X Y qx_peak, qy_peak, qz_peak, ||q||, I_peak
        results = np.zeros((n_xy_pos, 6), dtype=np.float64)

        measurement = master_h5[measurement_tpl.format(entries[0])]
        sample_x = measurement['adcX'][:]
        sample_y = measurement['adcY'][:]
        
        t_histo = 0.
        t_fit = 0.
        t_mask = 0.
        t_read = 0.
        t_dnsamp = 0.
        t_medfilt = 0.
        
        for image_idx in range(1000):
            
            if image_idx % 100 == 0:
                print('#{0}/{1}'.format(image_idx, n_xy))

            cumul = None

            i_sum = 0.
            for entry_idx, entry in enumerate(entries):

                t0 = time.time()
                img_data = master_h5[img_data_tpl.format(entry)]

                img = img_data[image_idx].astype(np.float64)

                t_read += time.time() - t0
                t0 = time.time()

                intensity = img.reshape(img_shape_1).\
                    sum(axis=sum_axis_1).reshape(img_shape_2).\
                    sum(axis=sum_axis_2) *\
                    avg_weight
                #intensity = xu.blockAverage2D(img, nav[0], nav[1], roi=roi)
                
                t_dnsamp += time.time() - t0
                t0 = time.time()

                intensity = medfilt(intensity, [3, 3])
                
                t_medfilt += time.time() - t0
                t0 = time.time()

                cumul = histogramnd_from_lut(intensity.reshape(-1),
                                             h_lut[entry_idx],
                                             shape=histo.shape,
                                             weighted_histo=cumul,
                                             dtype=np.float64)

                t_histo += time.time() - t0
                
            
            t0 = time.time()
            
            cumul[mask] = cumul[mask]/histo[mask]
            
            t_mask += time.time() - t0
            
            t0 = time.time()
        
            v0 = [1.0, qz.mean(), 1.0]
            qz_peak = leastsq(e_gauss_fit,
                              v0[:],
                              args=(qz_idx, (cumul.sum(axis=0)).sum(axis=0)),
                              maxfev=100000,
                              full_output=1)[0][1]
            v0 = [1.0, qy.mean(), 1.0] 
            qy_peak = leastsq(e_gauss_fit,
                              v0[:],
                              args=(qy_idx, (cumul.sum(axis=2)).sum(axis=0)),
                              maxfev=100000,
                              full_output=1)[0][1]
            v0 = [1.0, qx.mean(), 1.0] 
            qx_peak = leastsq(e_gauss_fit,
                              v0[:],
                              args=(qx_idx, (cumul.sum(axis=2)).sum(axis=1)),
                              maxfev=100000,
                              full_output=1)[0][1]
            i_peak = leastsq(e_gauss_fit,
                             v0[:],
                             args=(qx_idx, (cumul.sum(axis=2)).sum(axis=1)),
                             maxfev=100000,
                             full_output=1)[0][0]
            t_fit += time.time() - t0
            
            q = np.sqrt(qx_peak**2 + qy_peak**2 + qz_peak**2)
            results[image_idx] = (sample_x[image_idx],
                                  sample_y[image_idx],
                                  qx_peak,
                                  qy_peak,
                                  qz_peak,
                                  q,
                                  i_peak)
            #res = 'Img {0} : {1} {2} {3} {4} {5} {6} {7}\n'.format(image_idx,
                                                             #sample_x[image_idx],
                                                             #sample_y[image_idx],
                                                             #qx_peak,
                                                             #qy_peak,
                                                             #qz_peak,
                                                             #q,
                                                             #i_peak)
            
    tb = time.time()

    print('TOTAL', tb - ta)
    print('Read', t_read)
    print('Dn Sample', t_dnsamp)
    print('Medfilt', t_medfilt)
    print('Histo', t_histo)
    print('Mask', t_mask)
    print('Fit', t_fit)
    
    return results
