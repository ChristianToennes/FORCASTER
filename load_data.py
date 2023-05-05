import numpy as np
from skimage.measure import block_reduce
import struct
import os
import pydicom
import utils

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

    #f, gamma = i0.get_i0(prefix + r"\70kVp")
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

    skip = 2
    if images.shape[1] > 1000:
        skip *= 2 
    #if images.shape[1] < 600:
    #    skip = 1
    #skip = 2

    edges = 30
    oedges = 30

    #sel = np.zeros(images.shape, dtype=bool)
    #sel[:,20*skip:-20*skip:skip,20*skip:-20*skip:skip] = True
    #norm_images_gained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    #norm_images_ungained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    #sel[:,edges:-edges:skip,edges:-edges:skip] = True
    #sel_shape = (images.shape[0],images[0,edges:-edges].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,edges:-edges].shape[0])))
    #norm_images_gained = np.zeros(sel_shape, dtype=np.float32)
    #sel_shape = (images.shape[0],images[0,edges:-edges:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,edges:-edges:skip].shape[0])))
    #norm_images_ungained = np.zeros(sel_shape, dtype=np.float32)
    
    #gained_images = np.array([image*(1+gain/100) for image,gain in zip(images[sel].reshape(sel_shape), percent_gain)])
    
    #for i in range(len(fs)):
    #    norm_img = gained_images[i].reshape(norm_images_gained[0].shape)
    #    norm_images_gained[i] = norm_img
    #    norm_img = images[i][sel[i]].reshape(norm_images_gained[0].shape)
    #    norm_images_ungained[i] = norm_img

    oskip = 2
    #if images.shape[2] > 2000:
    #    oskip *= 2
    if edges <= 0:
        norm_images_gained = np.array([image*(1+gain/100) for image,gain in zip(images[:,::1,::1], percent_gain)])
        norm_images_ungained = np.array([image*(1+gain/100) for image,gain in zip(images[:,::1,::1], percent_gain)])
    else:
        norm_images_gained = np.array([image*(1+gain/100) for image,gain in zip(images[:,oedges:-oedges:1,oedges:-oedges:1], percent_gain)])
        norm_images_ungained = np.array([image*(1+gain/100) for image,gain in zip(images[:,edges:-edges:1,edges:-edges:1], percent_gain)])
        
    norm_images_gained = block_reduce(norm_images_gained, (1,oskip,oskip), np.mean)
    norm_images_ungained = block_reduce(norm_images_ungained, (1,skip,skip), np.median)
    print("skip", skip, oskip)
    #norm_images_ungained = images[:,edges:-edges:skip, edges:-edges:skip]
    
    gain = 2.30
    #gain = 1
    offset = 400
    use = fs>1
    while (np.max(norm_images_gained, axis=(1,2))[use] > (gain*fs[use])).all():
        gain += 1
    #gain = np.ones_like(fs)
    #offset = 0
    if False:
        for i in range(len(fs)):
            norm_img = norm_images_gained[i] / (offset + (gain*fs)[i])
            #norm_img = norm_images_gained[i] / (1.1*np.max(norm_images_gained[i]))
            if (norm_img==0).any():
                norm_img[norm_img==0] = np.min(norm_img[norm_img!=0])
            norm_images_gained[i] = -np.log(norm_img)

    return norm_images_gained, norm_images_ungained, offset+gain*fs, fs

def read_dicoms(indir, max_ims=np.inf):
    print("read dicoms", indir)
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
                
                
                #if ds[0x0021,0x1017].VR == "SL":
                #    sods = [np.array(ds[0x0021,0x1017].value)]*len(ts)
                #else:
                #    sods = [np.array(int.from_bytes(ds[0x0021,0x1017].value, "little", signed=True))]*len(ts)
                
                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)
                sods = [stparmdata["SOD_A"]] *len(ts)

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
                NumberOfFrames = ds.NumberOfFrames
                if len(ims) == 0:
                    #ims = ds.pixel_array[54:]
                    ims = ds.pixel_array[3:370]
                    NumberOfFrames = len(ims)
                else:
                    ims = np.vstack([ims, ds.pixel_array])

                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))

                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))
                #cs[:,2] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,2])
                #cs[:,1] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,1])

                #coord_systems.append(cs)
                #cs_interpol.append([cs, float(ds.PositionerPrimaryAngle), float(ds.PositionerSecondaryAngle), int(NumberOfFrames)])

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

                coord_systems = []
                for i in range(int(NumberOfFrames)):
                    coord_systems.append(cs)
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

    print("create numpy arrays")
    kvs = np.array(kvs)
    mas = np.array(mas)
    μas = np.array(μas)
    percent_gain = np.array(percent_gain)
    ims = np.array(ims)
    sids = np.array(sids)
    sods = np.array(sods)

        
        #ims = ims[:no_ims]
        #kvs = kvs[:no_ims]
        #mas = mas[:no_ims]
        #μas = μas[:no_ims]
        #percent_gain = percent_gain[:no_ims]
        #sids = sids[:no_ims]
        #sods = sods[:no_ims]

    thetas = np.array(thetas)
    phis = np.array(phis)
    
    coord_systems = np.array(coord_systems)

    cs = coord_systems
    
    if False:
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

    #if len(cs_interpol) > 0:
    #    coord_systems = np.array(cs_interpol)

    return ims_gained, ims_ungained, mas, kvs, angles, coord_systems, sids, sods
