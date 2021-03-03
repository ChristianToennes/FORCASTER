import numpy as np
import SimpleITK as sitk
import forcast
import utils
import i0
import os
import pydicom
import struct
import scipy.optimize
import astra
import time
import itertools

def normalize(images, mAs_array, kV_array, percent_gain):
    print("normalize images")
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

    #f, gamma = i0.get_i0(r"E:\output\70kVp")
    #kVs[70] = f

    fs = []
    gain = 3
    #gain = 1
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
        skip = 2
    if images.shape[1] < 600:
        skip = 1

    sel = np.zeros(images.shape, dtype=bool)
    sel[:,20*skip:-20*skip:skip,20*skip:-20*skip:skip] = True
    norm_images_gained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    norm_images_ungained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    gained_images = np.array([image*(1+gain/100) for image,gain in zip(images, percent_gain)])
    
    for i in range(len(fs)):
        norm_img = gained_images[i][sel[i]].reshape(norm_images_gained[0].shape)
        norm_images_gained[i] = norm_img
        norm_img = images[i][sel[i]].reshape(norm_images_gained[0].shape)
        norm_images_ungained[i] = norm_img

    gain = 2.30
    #gain = 1
    offset = 400
    use = fs>1
    while (np.max(norm_images_gained, axis=(1,2))[use] > (gain*fs[use])).all():
        gain += 1
    gain = np.ones_like(fs)
    offset = 0
    for i in range(len(fs)):
        norm_img = norm_images_gained[i] / (offset + (gain*fs)[i])
        norm_images_gained[i] = -np.log(norm_img)

    return norm_images_gained, norm_images_ungained, offset+gain*fs, fs

def read_dicoms(indir):
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
                if ds[0x0021,0x1017].VR == "SL":
                    sods = [np.array(ds[0x0021,0x1017].value)]*len(ts)
                else:
                    sods = [np.array(int.from_bytes(ds[0x0021,0x1017].value, "little", signed=True))]*len(ts)

                if ds[0x0019,0x1008].VR == "US":
                    percent_gain = [ds[0x0019,0x1008].value] * len(ts)
                else:
                    percent_gain = [float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False))] * len(ts)

                #sid = ds[0x0021,0x1031].value
                if ds[0x0021,0x100f].VR =="SL":
                    xray_info = np.array(ds[0x0021,0x100F].value)
                else:
                    xray_info = np.array(list(struct.iter_unpack("<l",ds[0x0021,0x100f].value))).flatten()
                kvs = xray_info[0::4]
                mas = xray_info[1::4]*xray_info[2::4]*0.001
                μas = xray_info[2::4]*0.001

            elif "NumberOfFrames" in dir(ds):
                ims = ds.pixel_array
                for i in range(int(ds.NumberOfFrames)):
                    ts.append(len(ts))
                    kvs.append(float(ds.KVP))
                    mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                    if ds[0x0021,0x1004].VR == "SL":
                        μas.append(float(ds[0x0021,0x1004].value)*0.001)
                    else:
                        μas.append(struct.unpack("<l", ds[0x0021,0x1004].value)[0]*0.001)
                    thetas.append(float(ds.PositionerPrimaryAngle))
                    phis.append(float(ds.PositionerSecondaryAngle))

                    stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                    cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))
                    cs[:,2] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,2])
                    cs[:,1] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,1])

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
            if len(ts)>0:
                break
        if len(ts)>0:
            break

    print("create numpy arrays")
    kvs = np.array(kvs)
    mas = np.array(mas)
    μas = np.array(μas)
    percent_gain = np.array(percent_gain)
    thetas = np.array(thetas)
    phis = np.array(phis)
    ims = np.array(ims)
    coord_systems = np.array(coord_systems)
    sids = np.array(sids)
    sods = np.array(sods)

    cs = coord_systems
    
    rv = cs[:,:,2]
    rv /= np.vstack((np.sum(rv, axis=-1),np.sum(rv, axis=-1),np.sum(rv, axis=-1))).T
    prim = np.arctan2(np.sqrt(rv[:,0]**2+rv[:,1]**2), rv[:,2])
    prim = np.arccos(rv[:,2] / np.sqrt(rv[:,0]**2+rv[:,1]**2+rv[:,2]**2) )
    sec = 0.5*np.pi-np.arctan2(rv[:,1], rv[:,0])
    prim = prim*180 / np.pi
    sec = sec * 180 / np.pi

    prim[cs[:,1,2]<0] *= -1
    sec[cs[:,1,2]<0] -= 180
    prim[np.bitwise_and(cs[:,2,2]<0, cs[:,1,1]>0) ] -= 180
    prim[cs[:,1,0]>0] -= 180

    #thetas = prim
    #phis = sec
    angles = np.vstack((thetas*np.pi/180.0, phis*np.pi/180.0, np.zeros_like(thetas))).T

    ims_gained, ims_ungained, i0s_gained, i0s_ungained = normalize(ims, μas, kvs, percent_gain)

    return ims_gained, ims_ungained, i0s_gained, i0s_ungained, angles, coord_systems, sids, sods


def main(proj_path, cbct_path):
    ims, ims_ungained, i0s, i0s_ungained, angles, coord_systems, sids, sods = read_dicoms(proj_path)

    origin, size, spacing, image = utils.read_cbct_info(cbct_path)
    real_image = utils.fromHU(sitk.GetArrayFromImage(image))

    real_image = real_image[::-1, ::-1]
    real_image = np.swapaxes(real_image, 1,2)
    real_image = np.swapaxes(real_image, 0, 2)

    detector_shape = np.array((1920,2480))
    detector_mult = np.floor(detector_shape / np.array(ims_ungained.shape[1:]))
    detector_shape = np.array(ims_ungained.shape[1:])
    detector_spacing = np.array((0.125, 0.125)) * detector_mult

    cali = {}
    cali['feat_thres'] = 80
    cali['iterations'] = 50
    cali['confidence_thres'] = 0.025
    cali['relax_factor'] = 0.3
    cali['match_thres'] = 60
    cali['max_ratio'] = 0.9
    cali['max_distance'] = 20
    cali['outlier_confidence'] = 85

    geo, (prims, secs), _ = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods, sids-sods, 1.2/np.min(spacing))

    #print(angles, prims, secs)

    input_sino = np.swapaxes(ims,0,1)
    sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join("recos", "forcast_input.nrrd"))

    Ax = utils.Ax_geo_astra(real_image.shape, real_image)
    sino = Ax(geo)
    sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino.nrrd"))

    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], geo['Vectors'][0:1])
    proj_d = forcast.Projection_Preprocessing(Ax(geo_d))[:,0]
    real_img = forcast.Projection_Preprocessing(ims[0])
    cur = forcast.roughRegistration(np.array([0,0,0,0,0.0]), real_img, proj_d, {'feat_thres': cali['feat_thres']}, geo['Vectors'][0])
    vec = forcast.applyChange(geo['Vectors'][0], cur)
    vecs = np.array([vec])
    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
    proj_d = forcast.Projection_Preprocessing(Ax(geo_d))
    sitk.WriteImage(sitk.GetImageFromArray(proj_d), os.path.join("recos", "forcast_rough.nrrd"))

    for i in range(len(ims)):
        print("Projection ", i)
        funs = []
        with open('stats.csv','w') as f:
            good_values= [
                [5,7.33333333,0.05],
                [5,17,0.05],
            ]
            for e,m1,m2 in good_values:
                try:
                    perftime = time.perf_counter()
                    bfgs_vecs, fun, err = forcast.bfgs(i, ims, real_image, cali, geo, real_image.shape, Ax, np.array([e,e*m1,e,e*m2,e*m2]))
                    perftime = time.perf_counter()-perftime
                    print(int(fun), int(err), e, m1, m2, perftime)
                    funs.append([e,m1,m2,fun,err,perftime])
                    f.write(",".join([str(e) for e in [e,m1,m2,fun,err,perftime]])+"\n")
                    if fun < 12000:
                        bfgs_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([bfgs_vecs]))
                        sino = Ax(bfgs_geo)
                        #sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join("recos", "forcast_input.nrrd"))
                        sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino_bfgs--"+str(fun)+"--"+str(e)+"_"+str(m1)+"_"+str(m2)+".nrrd"))
                except Exception as ex:
                    print(ex)
            for xy in itertools.chain(np.linspace(0.1,1,10), np.linspace(1,10,10)):
                #for m1 in np.linspace(3,100, 5):
                for z in itertools.chain(np.linspace(0.1,1,10), np.linspace(1,10,10)):
                    for r in itertools.chain(np.linspace(0.01,0.1,10), np.linspace(0.1,10,10)):
                        #print(e, m)
                        try:
                            perftime = time.perf_counter()
                            bfgs_vecs, fun, err = forcast.bfgs(i, ims, real_image, cali, geo, real_image.shape, Ax, np.array([xy,xy,z,r,r]))
                            perftime = time.perf_counter()-perftime
                            print(int(fun), int(err), xy, z, r, perftime)
                            funs.append([e,m1,m2,fun,err,perftime])
                            f.write(",".join([str(e) for e in [xy,z,r,fun,err,perftime]])+"\n")
                            if fun < 12000:
                                bfgs_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([bfgs_vecs]))
                                sino = Ax(bfgs_geo)
                                #sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join("recos", "forcast_input.nrrd"))
                                sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino_bfgs--"+str(fun)+"--"+str(xy)+"_"+str(z)+"_"+str(r)+".nrrd"))
                        except Exception as ex:
                            print(ex)
        #with open('stats.csv','w') as f:
            #f.writelines([",".join([str(e) for e in line])+"\n" for line in funs])
        my_vecs = forcast.FORCAST(i, ims, real_image, cali, geo, real_image.shape, np.array([2.3,2.3*3,2.3,2.3*0.1,2.3*0.1]), Ax)
        my_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([my_vecs]))
        sino = Ax(my_geo)
        sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino_my.nrrd"))

    rec = utils.FDK_astra(real_image.shape, geo)(np.swapaxes(ims, 0,1))
    rec = np.swapaxes(rec, 0, 2)
    rec = np.swapaxes(rec, 1,2)
    rec = rec[::-1, ::-1]
    sitk.WriteImage(sitk.GetImageFromArray(rec), os.path.join("recos", "forcast_reco.nrrd"))


if __name__ == "__main__":
    cbct_path = r"E:\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    proj_path = 'E:\\output\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD'
    main(proj_path, cbct_path)