import numpy as np
import nibabel as nb


def py_SurfStatAvVol(filenames, fun = np.add, replace_nan = np.nan):

    #filenames = file name with extension .mnc, .img, .nii or .brik as above 
    #        (n=1), or n x 1 cell array of file names.

    n = np.shape(filenames)[0]

    file_01 = filenames[0][0]

    if file_01.endswith('.nii') or file_01.endswith('.nii.gz') or \
        file_01.endswith('.img'):

        if n == 1:
            data_i = np.array(nb.load(file_01).get_fdata())
            m = 1
        else:
            for i in range(0, n):
                d = nb.load(filenames[i][0])
                data = np.array(d.get_fdata())

                if replace_nan != np.nan:
                    data[np.isnan(data)] = replace_nan

                if i == 0:
                    data_i = data
                    m = 1
                else:
                    data_i = fun(data_i, data)
                    m = fun(m, 1)

    data_i = data_i/float(m)

    vol = {}
    vol['lat'] = np.ones(d.shape[0:3], dtype=bool)
    vol['vox'] = np.array(d.header.get_zooms()[0:3])
    vol['origin'] = d.header['origin'][0:3]
    
    return data, vol

