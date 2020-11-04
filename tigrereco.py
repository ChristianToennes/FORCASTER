from __future__ import print_function
from __future__ import division
import tigre
import copy
import tigre.algorithms as algs
import numpy as np
import time
import sys
from tigre.demos.Test_data import data_loader
from matplotlib import pyplot as plt
from tigre.utilities.Measure_Quality import Measure_Quality
import pydicom
from reco import load_image, normalize
import SimpleITK as sitk

def reco(s, niter = 20):
    #img = pydicom.read_file("InitialCBCT.dcm")
    #proj_data = np.moveaxis(img.pixel_array, -1, 0)
    #proj = img.pixel_array

    #print("projection data read", proj.shape)

    #angles = np.linspace(0, np.pi, 180,False)
    #angles = np.array(img[0x0018,0x1520].value)*np.pi / 180
    #print("angles", angles)
    #angles -= np.min(angles)
    #print("angles", angles)
    #angles2 = img[0x0018,0x1521].value

    #dist_source_detector = int(img[0x0018,0x1110].value)
    #dist_source = int.from_bytes(img[0x0021,0x1017].value, "little")
    #dist_detector = (dist_source_detector - dist_source)
    #print("dist source", dist_source, "dist_detector", dist_detector)
    
    if s == 0:
        (raw_img, angles1, angles2, dist_source, dist_detector, kv, ms, As, mAs) = load_image(False)
        prefix = "trajopti_"
    elif s == 1:
        (raw_img, angles1, angles2, dist_source, dist_detector, kv, ms, As, mAs) = load_image(False, "TrajTomo")
        prefix = "trajtomo_"
    else:
        (raw_img, angles1, angles2, dist_source, dist_detector, kv, ms, As, mAs) = load_image(True)
        prefix = "cbct_"

    proj = np.array(normalize(raw_img, mAs, kv), dtype=np.float32)

    #geo1 = tigre.geometry(mode='cone', high_quality=False, default=True)
    geo = tigre.geometry(mode='cone', nVoxel=np.array([512,512,512]),default=True)
    geo.nDetector = np.array(proj.shape[1:])
    geo.dDetector = np.array([0.154, 0.154])               # size of each pixel            (mm)
    geo.sDetector = geo.dDetector * geo.nDetector
    geo.DSD = np.mean(dist_source) + np.mean(dist_detector)
    geo.DSO = np.mean(dist_source)
    geo.sVoxel = np.array([150,150,150])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    #DDO = np.min(dist_detector+dist_source)
    #print(geo)

    angles = np.vstack([angles1, angles2, np.zeros_like(angles1)]).T
    #print(angles1.shape, angles2.shape, angles.shape)
    #print(proj.shape)
    
    print("start fdk")
    fdkout = algs.fdk(proj,geo,angles)
    sitk.WriteImage(sitk.GetImageFromArray(fdkout), prefix+"reco_tigre_fdk.nrrd")
    print("start ossart")
    sirtout = algs.ossart(proj,geo,angles,niter,blocksize=20)
    sitk.WriteImage(sitk.GetImageFromArray(sirtout), prefix+"reco_tigre_ossart.nrrd")
    print("start cgls")
    cglsout = algs.cgls(proj,geo,angles,niter)
    sitk.WriteImage(sitk.GetImageFromArray(cglsout), prefix+"reco_tigre_cgls.nrrd")
    print("start fista")
    fistaout = algs.fista(proj,geo,angles,niter,hyper=2.e4)
    sitk.WriteImage(sitk.GetImageFromArray(fistaout), prefix+"reco_tigre_fista.nrrd")
    niter=5
    print("start asd pocs")
    asdpocsout = algs.asd_pocs(proj,geo,angles,niter)
    sitk.WriteImage(sitk.GetImageFromArray(asdpocsout), prefix+"reco_tigre_asd_pocs.nrrd")

    #plt.figure(1)
    #plt.gray()
    #plt.title("fdk")
    #plt.imshow(fdkout[geo.nVoxel[0]//2])
    #plt.figure(2)
    #plt.gray()
    #plt.title("ossart")
    #plt.imshow(sirtout[geo.nVoxel[0]//2])
    #plt.figure(3)
    #plt.gray()
    #plt.title("cgls")
    #plt.imshow(cglsout[geo.nVoxel[0]//2])
    #plt.figure(4)
    #plt.gray()
    #plt.title("asd pocs")
    #plt.imshow(asdpocsout[geo.nVoxel[0]//2])
    #plt.figure(5)
    #plt.gray()
    #plt.title("fista")
    #plt.imshow(fistaout[geo.nVoxel[0]//2])
    #plt.figure(6)
    #plt.gray()
    #plt.title("target")
    #target = pydicom.read_file("100252.000000_197.dcm").pixel_array
    #plt.imshow(target)
    #plt.show()

reco(0)
#reco(1)
reco(2)