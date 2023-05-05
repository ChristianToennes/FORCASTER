import numpy as np
import os
import scipy.ndimage
import pydicom

def i0_est(real_img, proj_img):
    filt = np.zeros_like(real_img, dtype=bool)
    filt[filt.shape[0]//3:-filt.shape[0]//3,filt.shape[1]//3:-filt.shape[1]//3] = True
    real = float(np.mean(real_img[filt]))
    p = scipy.ndimage.zoom(proj_img, np.array(real_img.shape)/np.array(proj_img.shape))
    proj = np.mean(p[filt])
    i0 = real * np.exp(proj)
    
    #i0 = i0*1.4
    i0 += 400
    return i0

def i0_data(skip, i_edges):
    if os.path.exists("F:\\output"):
        prefix = r"F:\output"
    elif os.path.exists(r"D:\lumbal_spine_13.10.2020\output"):
        prefix = r"D:\lumbal_spine_13.10.2020\output"
    else:
        prefix = r".\output"

    path = os.path.join(prefix, '70kVp')

    kvs = []
    mas = []
    ims = []

    for root, _dirs, files in os.walk(path):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            ds = pydicom.dcmread(path)
            edges = skip*(ds.pixel_array.shape[1]//skip-i_edges)//2
            #print(ds.pixel_array.shape,i_edges, edges)
            if edges <= 0:
                ims.append(np.mean(ds.pixel_array[:,::skip, ::skip], axis=0))
            else:
                ims.append(np.mean(ds.pixel_array[:,edges:-edges:skip, edges:-edges:skip], axis=0))
            kvs.append(float(ds.KVP))
            mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)

    kvs = np.array(kvs)
    mas = np.array(mas)
    ims = np.array(ims)

    print(ims.shape)
    return ims, mas, kvs

def i0_interpol(ims, mas, target):
    res = np.zeros_like(ims[0])

    for i in range(ims.shape[1]):
        for j in range(ims.shape[2]):
            p = np.polyfit(np.array(mas), ims[:,i,j], 1)
            res[i,j] = np.polyval(p, target)

    return res
