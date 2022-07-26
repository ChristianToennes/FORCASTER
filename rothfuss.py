import numpy as np
import SimpleITK as sitk
from simple_cal import *
import cal
import utils
from utils import bcolors
import os
import pydicom
import struct
import astra
import time
import cv2
from feature_matching import Projection_Preprocessing
import matplotlib.pyplot as plt

data_path = [r"D:\rothfuss\ProejctionData\Artis zeego CUBEX41\TESSERAKT_TEST_21_10_28-09_22_24-DST-1_3_12_2_1107_5_4_5_160969\__20211028_093110_660000", "6SDCT_BODY_0001", "DCT_BODY_NAT_FILL_FULL_HU_NORMAL_[AX3D]_0001"]


def normalize(images, mAs_array, kV_array, percent_gain):
    kVs = {}
    kVs[40] = (20.4125, 677.964)
    kVs[50] = (61.6163, 686.4824)
    kVs[60] = (138.4021, 684.1844)
    kVs[70] = (250.8008, 691.9573)
    kVs[80] = (398.963, 701.1038)
    kVs[90] = (586.5949, 711.416)
    kVs[100] = (794.5124, 729.8813)
    kVs[109] = (1006.1, 750.0054)
    kVs[120] = (1252.2, 791.9865)
    kVs[125] = (1404.2202, 796.101)

    fs = []
    gain = 3
    for mAs, kV in zip(mAs_array, kV_array):
        if kV in kVs:
            f = np.polyval(kVs[kV], mAs)
        else:
            kVs_keys = np.array(list(sorted(kVs.keys())))
            if kV < kVs_keys[0]:
                f = np.polyval(kVs[kVs_keys[0]], mAs)
            elif kV > kVs_keys[-1]:
                f = np.polyval(kVs[kVs_keys[-1]], mAs)
            else:
                i1, i2 = np.argsort(np.abs(kVs_keys-kV))[:2]
                f1 = np.polyval(kVs[kVs_keys[i1]], mAs)
                f2 = np.polyval(kVs[kVs_keys[i2]], mAs)
                d1 = np.abs(kVs_keys[i1]-kV)*1.0
                d2 = np.abs(kVs_keys[i2]-kV)*1.0
                f = f1*(1.0-(d1/(d1+d2))) + f2*(1.0-(d2/(d1+d2)))
        fs.append(f)
    
    fs = np.array(fs).flatten()

    skip = 4
    if images.shape[1] < 1000:
        skip = 1

    edges = 30

    oskip = 1
    if images.shape[2] > 2000:
        oskip = 4
        edges *= 4
    if edges <= 0:
        norm_images_gained = np.array([image*(1+gain/100) for image,gain in zip(images[:,::oskip,::oskip], percent_gain)])
        norm_images_ungained = np.array([image*(1+gain/100) for image,gain in zip(images[:,::skip,::skip], percent_gain)])
    else:
        norm_images_gained = np.array([image*(1+gain/100) for image,gain in zip(images[:,edges:-edges:oskip,edges:-edges:oskip], percent_gain)])
        norm_images_ungained = np.array([image*(1+gain/100) for image,gain in zip(images[:,edges:-edges:skip,edges:-edges:skip], percent_gain)])
    
    gain = 2.30
    offset = 400
    use = fs>1
    while (np.max(norm_images_gained, axis=(1,2))[use] > (gain*fs[use])).all():
        gain += 1
    if False:
        for i in range(len(fs)):
            norm_img = norm_images_gained[i] / (offset + (gain*fs)[i])
            #norm_img = norm_images_gained[i] / (1.1*np.max(norm_images_gained[i]))
            if (norm_img==0).any():
                norm_img[norm_img==0] = np.min(norm_img[norm_img!=0])
            norm_images_gained[i] = -np.log(norm_img)

    return norm_images_gained, norm_images_ungained, offset+gain*fs, fs

def read_dicoms(indir, max_ims=np.inf):
    print("read dicoms")
    kvs = []
    mas = []
    μas = []
    ts = []
    thetas = []
    phis = []
    prims = []
    secs = []
    ims = []
    percent_gain = []
    coord_systems = []
    cs_interpol = []
    sids = []
    sods = []

    #sid = []
    for root, _dirs, files in os.walk(indir):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            #read DICOM files
            ds = pydicom.dcmread(path)
            if "PositionerPrimaryAngleIncrement" in dir(ds):
                ims = ds.pixel_array
                ts = list(range(len(ims)))
                thetas = np.array(ds.PositionerPrimaryAngleIncrement)
                phis = np.array(ds.PositionerSecondaryAngleIncrement)
                
                if ds[0x0021,0x1059].VR == "FL":
                    cs = np.array(ds[0x0021,0x1059].value).reshape((len(ts), 3, 4))
                    coord_systems = np.array(ds[0x0021,0x1059].value).reshape((len(ts), 3, 4))
                else:
                    cs = np.array(list(struct.iter_unpack("<f", ds[0x0021,0x1059].value))).reshape((len(ts), 3, 4))
                    coord_systems = np.array(list(struct.iter_unpack("<f", ds[0x0021,0x1059].value))).reshape((len(ts), 3, 4))
                
                if ds[0x0021,0x1031].VR == "SS":
                    sids = np.array(ds[0x0021,0x1031].value)*0.1
                else:
                    sids = np.array(list(struct.iter_unpack("<h", ds[0x0021,0x1031].value))).flatten()*0.1
                
                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)
                sods = [stparmdata["SOD_A"]] *len(ts)

                if ds[0x0019,0x1008].VR == "US":
                    percent_gain = [ds[0x0019,0x1008].value] * len(ts)
                else:
                    percent_gain = [float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False))] * len(ts)

                if ds[0x0021,0x100f].VR =="SL":
                    xray_info = np.array(ds[0x0021,0x100F].value)
                else:
                    xray_info = np.array(list(struct.iter_unpack("<l",ds[0x0021,0x100f].value))).flatten()
                kvs = xray_info[0::4]
                mas = xray_info[1::4]*xray_info[2::4]*0.001
                μas = xray_info[2::4]*0.001

            elif "NumberOfFrames" in dir(ds):
                if len(ims) == 0:
                    ims = ds.pixel_array
                else:
                    ims = np.vstack([ims, ds.pixel_array])
                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))
                coord_systems.append(cs)
                cs_interpol.append([cs, float(ds.PositionerPrimaryAngle), float(ds.PositionerSecondaryAngle), int(ds.NumberOfFrames)])
                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))

                rv = cs[:,2]
                rv /= np.sum(rv**2, axis=-1)
                prim = np.arctan2(np.sqrt(rv[0]**2+rv[1]**2), rv[2])
                prim = np.arccos(rv[2] / np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2) )
                prim = prim*180 / np.pi
                

                if cs[1,2]<0:
                    prim *= -1

                if (prim > 50 and prim < 135) or (prim <-45 and prim > -135):
                    sec = np.arctan2(rv[1], rv[0])
                    sec = sec * 180 / np.pi
                    if cs[1,2]<0:
                        sec *= -1
                    sec -= 90
                else:
                    sec = np.arctan2(rv[2], rv[0])
                    sec = sec * 180 / np.pi - 90

                prims.append(prim)
                secs.append(sec)

                for i in range(int(ds.NumberOfFrames)):
                    ts.append(len(ts))
                    kvs.append(float(ds.KVP))
                    mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                    if ds[0x0021,0x1004].VR == "SL":
                        μas.append(float(ds[0x0021,0x1004].value)*0.001)
                    else:
                        μas.append(struct.unpack("<l", ds[0x0021,0x1004].value)[0]*0.001)
                    
                    sids.append(np.array(stparmdata["SID_A"]))
                    sods.append(np.array(stparmdata["SOD_A"]))
                    
                    if ds[0x0019,0x1008].VR == "US":
                        percent_gain.append(ds[0x0019,0x1008].value)
                    else:
                        percent_gain.append(float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False)))

            elif "PositionerPrimaryAngle" in dir(ds):
                ts.append(len(ts))
                kvs.append(float(ds.KVP))
                mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                if ds[0x0021,0x1004].VR == "SL":
                    μas.append(float(ds[0x0021,0x1004].value)*0.001)
                else:
                    μas.append(struct.unpack("<l", ds[0x0021,0x1004].value)[0]*0.001)
                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))
                ims.append(ds.pixel_array)

                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))

                coord_systems.append(cs)
                sids.append(np.array(stparmdata["SID_A"]))
                sods.append(np.array(stparmdata["SOD_A"]))

                rv = cs[:,2]
                rv /= np.sum(rv**2, axis=-1)
                prim = np.arctan2(np.sqrt(rv[0]**2+rv[1]**2), rv[2])
                prim = np.arccos(rv[2] / np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2) )
                prim = prim*180 / np.pi

                if cs[1,2]<0:
                    prim *= -1

                if (prim > 50 and prim < 135) or (prim <-45 and prim > -135):
                    sec = np.arctan2(rv[1], rv[0])
                    sec = sec * 180 / np.pi
                    if cs[1,2]<0:
                        sec *= -1
                    sec -= 90
                else:
                    sec = np.arctan2(rv[2], rv[0])
                    sec = sec * 180 / np.pi - 90

                prims.append(prim)
                secs.append(sec)

                if ds[0x0019,0x1008].VR == "US":
                    percent_gain.append(ds[0x0019,0x1008].value)
                else:
                    percent_gain.append(float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False)))
            
            del ds
            if len(ts)>=max_ims:
                break
        if len(ts)>=max_ims:
            break

    kvs = np.array(kvs)
    mas = np.array(mas)
    μas = np.array(μas)
    percent_gain = np.array(percent_gain)
    ims = np.array(ims)
    sids = np.array(sids)
    sods = np.array(sods)

    thetas = np.array(thetas)
    phis = np.array(phis)
    
    coord_systems = np.array(coord_systems)

    cs = coord_systems
    
    angles = np.vstack((thetas*np.pi/180.0, phis*np.pi/180.0, np.zeros_like(thetas))).T

    ims_gained, ims_ungained, i0s_gained, i0s_ungained = normalize(ims, μas, kvs, percent_gain)

    #if len(cs_interpol) > 0:
    #    coord_systems = np.array(cs_interpol)

    return ims_gained, ims_ungained, mas, kvs, angles, coord_systems, sids, sods


def load_data():
    #ims_gained, ims_ungained, mas, kvs, angles, coord_systems, sids, sods = read_dicoms(data_path[0]+"/"+data_path[1])
    ims_gained, ims_ungained, mas, kvs, angles, coord_systems, sids, sods = read_dicoms(data_path[0]+"/DR_OVERVIEW_0002")

    origin, size, spacing, image = utils.read_cbct_info(data_path[0]+"/"+data_path[2])
    real_image = utils.fromHU(sitk.GetArrayFromImage(image))

    return real_image, ims_ungained, angles, coord_systems, sids, sods, spacing


if __name__ == "__main__":
    real_image, ims_ungained, angles, coord_systems, sids, sods, spacing = load_data()
    print(angles, coord_systems)

    i = -1
    
    detector_shape = np.array((1920,2480))
    detector_mult = int(np.floor(detector_shape[0] / ims_ungained.shape[1]))
    
    detector_shape = np.array(ims_ungained.shape[1:])
    detector_spacing = np.array((0.125, 0.125)) * detector_mult

    Ax = utils.Ax_param_asta(real_image.shape, detector_spacing, detector_shape, sods[i], sids[i]-sods[i], 1.2/np.min(spacing), real_image)

    #if coord_systems.shape[1] == 4:
    #    coord_systems, thetas, phis, params = interpol_positions(coord_systems, Ax, ims, detector_spacing, detector_shape, sods, sids-sods, 1.2/np.min(spacing))
    #    params = params[skip]
    #coord_systems = coord_systems

    Ax_gen = (real_image.shape, detector_spacing, detector_shape, sods[i], sids[i]-sods[i], 1.2/np.min(spacing), real_image)
    geo = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods[i], sids[i]-sods[i], 1.2/np.min(spacing))
    coords_from_angles = utils.angles2coord_system(angles)
    geo_from_angles = utils.create_astra_geo_coords(coords_from_angles, detector_spacing, detector_shape, sods[i], sids[i]-sods[i], 1.2/np.min(spacing))
    r = utils.rotMat(90, [1,0,0]).dot(utils.rotMat(-90, [0,0,1]))
    
    params = np.zeros((len(geo['Vectors']), 3, 3), dtype=float)
    params[:,1] = np.array([r.dot(v) for v in geo['Vectors'][:, 6:9]])
    params[:,2] = np.array([r.dot(v) for v in geo['Vectors'][:, 9:12]])
    
    config = dict(default_config)
    config["Ax"] = Ax
    config["Ax_gen"] = Ax_gen
    config["method"] = 3
    config["name"] = "rothfuss"
    config["real_cbct"] = real_image
    config["outpath"] = r"D:\rothfuss\ProejctionData\out"
    config["estimate"] = False

    real_img = cal.Projection_Preprocessing(ims_ungained[i])
    config["real_img"] = real_img
    cur = np.array(params[i])

    config["data_real"] = findInitialFeatures(real_img, config)

    print(cur)
    config["it"] = 3
    cur = correctXY(cur, config)
    cur = correctZ(cur, config)
    cur = correctXY(cur, config)

    print(cur)