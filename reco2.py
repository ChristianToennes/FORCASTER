import SimpleITK as sitk
import astra
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import os.path
import tigre
import time
import i0
import mlem
import utils
import piccs
import scipy.ndimage
import scipy.signal
import struct

def conv_time(time):
    h = float(time[:2])*60*60
    m = float(time[2:4])*60
    s = float(time[4:])
    return h+m+s


def normalize(images, mAs_array, kV_array, gammas, window_center, window_width, percent_gain):
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

    #kVs[70] = (98.2053, 232.7655)
    #kVs[70] = (86.04351234294207, 20.17212116766863)
    #kVs[70] = (-3.27759476, 264.10304478, 602.69536172)
    
    f, gamma = i0.get_i0(r".\output\70kVp")
    #f, gamma = i0.get_i0(r".\output\CKM\I0 Daten")
    #f, gamma = i0.get_i0(r".\output\CKM\CircTomo\20201207-094635.313000-P16_Card_HD")
    #print(f)
    kVs[70] = f

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

    #print(images.shape)
    sel = np.zeros(images.shape, dtype=bool)
    #sel[:,::skip,::skip] = True
    if skip == 2:
        for i in range(sel.shape[0]):
            s = 600*skip
            if i < 223:
                l = 20*skip
            elif i >= 223 and i <= 323:
                l = 20*skip-int(np.round(skip*(i-223)/5))
            else:
                l = 0
            r = (s+l)
            #print(s, l, r, sel[0,0,:l].shape[0], sel[0,0,r:].shape[0], sel[0,0,l:r].shape[0])
            sel[i,20*skip:-20*skip:skip,l:r:skip] = True
            #sel[:,:,:l] = False
    else:
        sel[:,20*skip:-20*skip:skip,20*skip:-20*skip:skip] = True
    #print(s,l,r)
    #skip = 1
    #norm_images = images / fs
    #norm_images = np.zeros(images[:, (10*skip):-(10*skip):skip,(10*skip):-(10*skip):skip].shape, dtype=np.float32)
    norm_images = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    #print(norm_images.shape, sel.shape)
    #print(np.size(norm_images)/images.shape[0]/(images.shape[1]/skip), np.count_nonzero(sel)/images.shape[0]/(images.shape[1]/skip))
    #print(np.count_nonzero(sel, axis=(1,2)))
    #minWindow = window_center - window_width / 2
    #windowScale = window_width / 4096
    #maxValue = minWindow + window_width
    #wv = []
    #igamma = inverse_lut(gamma)

    gained_images = np.array([image*(1+gain/100) for image,gain in zip(images, percent_gain)])
    
    for i in range(len(fs)):
        #norm_img = gained_images[i, (10*skip):-(10*skip):skip,(10*skip):-(10*skip):skip]
        norm_img = gained_images[i][sel[i]].reshape(norm_images[0].shape)
        #print(norm_img.shape)
        #norm_img = scipy.ndimage.gaussian_filter(gained_images[i], 3)[(10*skip):-(10*skip):skip,(10*skip):-(10*skip):skip]
        #norm_img = scipy.ndimage.median_filter(gained_images[i], skip)[(10*skip):-(10*skip):skip,(10*skip):-(10*skip):skip]
        norm_images[i] = norm_img

    gain = 2.30
    #gain = 1
    offset = 400
    use = fs>1
    while (np.max(norm_images, axis=(1,2))[use] > (gain*fs[use])).all():
        gain += 1

    #gain += 1
    print("used gain: ", gain)
    for i in range(len(fs)):
        #gain = fs[i] / np.max(norm_images[i])
        norm_img = norm_images[i] / (offset + gain*fs[i])
        norm_images[i] = -np.log(norm_img)

    #print(np.mean(fs), np.median(fs), np.max(fs), np.min(fs))
    #print(np.mean(mAs_array), np.median(mAs_array), np.max(mAs_array), np.min(mAs_array))
    #print(np.mean(gained_images), np.median(gained_images), np.max(gained_images), np.min(gained_images))
    #print(np.mean(images), np.median(images), np.max(images), np.min(images))
    #print(np.mean(norm_images), np.median(norm_images), np.max(norm_images), np.min(norm_images))
    return norm_images, fs

def filter_images(ims, ts, angles, mas):
    print("filter images")  
    filt = np.zeros(ts.shape, dtype=bool)

    for i, (angle, ma) in enumerate(zip(angles, mas)):
        if ma <= np.min(mas[np.bitwise_and(angles[:,0]==angle[0], angles[:,1]==angle[1])]):
            filt[i] = True
    
    diff = np.linspace(angles[0,0],angles[-1,0],len(angles))-angles[:,0]
    diff[1:] = angles[:-1,0]-angles[1:,0]-0.4
    filt = np.bitwise_or(np.bitwise_or(diff>0.1, diff<0.1), filt)
    return filt

def inverse_lut(lut):
    ilut = np.zeros_like(lut)
    for i in range(len(lut)):
        pos = np.argmin(np.abs(lut-i))
        if np.isscalar(pos):
            ilut[i] = pos
        else:
            ilut[i] = pos[0]
    return ilut

def read_dicoms(indir, prefix, reg_angles=False):
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
    #gammas = []
    gamma_in = []
    gamma_out = []
    window_center = []
    window_width = []
    water_value = []
    percent_gain = []
    coord_systems = []
    sids = []
    sods = []
    drot = []

    #sid = []
    for root, dirs, files in os.walk(indir):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            #read DICOM files
            ds = pydicom.dcmread(path)

            if "PositionerPrimaryAngleIncrement" in dir(ds):
                #ts.append(conv_time(ds.AcqusitionTime))
                #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
                ims = ds.pixel_array
                #if (ds.BitsStored == 16):
                #    ims = np.array(ims*(2.0**14/2.0**16), dtype=int)
                ts = list(range(len(ims)))
                thetas = np.array(ds.PositionerPrimaryAngleIncrement)
                phis = np.array(ds.PositionerSecondaryAngleIncrement)
                window_center = [float(ds.WindowCenter)] * len(ts)
                window_width = [float(ds.WindowWidth)] * len(ts)

                #if ds[0x0021,0x1028].VR == "SQ":
                #gammas = [np.array(ds[0x0021,0x1028][0][0x0021,0x1042].value)] * len(ts)
                    

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

                if ds[0x0021,0x1071].VR == "DS":
                    drot = np.array(ds[0x0021,0x1071].value)
                else:
                    drot = np.array(list(struct.iter_unpack("<f", ds[0x0021,0x1071].value)))


                #plt.plot(np.arange(len(thetas)), prim)
                #plt.plot(np.arange(len(thetas)), thetas)
                #plt.show()
                #plt.plot(np.arange(len(phis)), sec)
                #plt.plot(np.arange(len(phis)), phis)
                #plt.show()
                #exit(0)

                #thetas = prim
                #phis = sec

                #e1 = thetas-prim
                #e2 = phis-sec
                #print(np.mean(e1), np.median(e1), np.max(e1), np.argmax(e1>5))
                #print(np.mean(e2), np.median(e2), np.max(e2), np.argmax(e2>5))

                #plt.figure()
                #diff = np.linspace(thetas[0],thetas[-1],len(thetas))-thetas
                #plt.plot(np.arange(len(diff)), diff)
                #diff = thetas[:-1]-thetas[1:]-0.4
                #plt.plot(np.arange(len(diff)), diff)
                #plt.show()
                #plt.close()

                if ds[0x021,0x1010].VR == "US":
                    gamma_in = [float(ds[0x021,0x1010].value)] * len(ts)
                else:
                    gamma_in = [float(int.from_bytes(ds[0x021,0x1010].value, "little", signed=False))] * len(ts)
                if ds[0x021,0x1011].VR == "US":
                    gamma_out = [float(ds[0x021,0x1011].value)] * len(ts)
                else:
                    gamma_out = [float(int.from_bytes(ds[0x021,0x1011].value, "little", signed=False))] * len(ts)
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

                #thetas2 = np.array(ds[0x0021,0x1068].value)
                #phis2 = np.array(ds[0x0021,0x1072].value)/100

                #water_value = [float(ds[0x0021,0x1049].value)]*len(ts)

                #print(thetas-thetas2/100)
                #print(phis-phis2)
                #angulation = ds[0x0021,0x105D].value
                #orbital = ds[0x0021,0x105E].value
            elif "PositionerPrimaryAngle" in dir(ds):
                #ts.append(conv_time(ds.AcquisitionTime))
                #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
                ts.append(len(ts))
                kvs.append(float(ds.KVP))
                mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                μas.append(float(ds[0x0021,0x1004].value)*0.001)
                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))

                if ds[0x0021,0x1071].VR == "DS":
                    drot.append(np.array(ds[0x0021,0x1071].value))
                else:
                    drot.append(np.array(list(struct.iter_unpack("<d", ds[0x0021,0x1071].value))))

                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))
                #print(stparmdata["ISO_WORLD_X_A"],stparmdata["ISO_WORLD_Y_A"],stparmdata["ISO_WORLD_Z_A"], cs[:,3])
                #cs[0,3] = stparmdata["ISO_WORLD_X_A"]*0.1 - cs[0,3]
                #cs[1,3] = stparmdata["ISO_WORLD_Y_A"]*0.1 - cs[1,3]
                #cs[2,3] = stparmdata["ISO_WORLD_Z_A"]*0.1 - cs[2,3]
                coord_systems.append(cs)
                sids.append(np.array(stparmdata["SID_A"]))
                sods.append(np.array(stparmdata["SOD_A"]))

                rots = np.arccos((cs[0,0]+cs[1,1]+cs[2,2]-1)/2)
                e1s = (cs[2,1]-cs[1,2]) / (2*np.sin(rots))
                e2s = (cs[0,1]-cs[2,1]) / (2*np.sin(rots))
                e3s = (cs[1,0]-cs[0,1]) / (2*np.sin(rots))
                ls = np.sqrt(e1s**2 + e2s**2 + e3s**2)
                es = np.vstack((e1s/ls, e2s/ls, e3s/ls)).T
                
                rv = cs[:,2]
                rv /= np.sum(rv**2, axis=-1)
                prim = np.arctan2(np.sqrt(rv[0]**2+rv[1]**2), rv[2])
                prim = np.arccos(rv[2] / np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2) )
                prim = prim*180 / np.pi
                

                if cs[1,2]<0:
                    prim *= -1
                    #sec -= 180
                #if cs[2,2]<0 and cs[1,1]>0:
                #    prim -= 180
                #if cs[1,0]>0:
                #    prim -= 180
                #if cs[1,2]>0 and cs[2,1]<0:
                #    prim += 360
                if (prim > 50 and prim < 135) or (prim <-45 and prim > -135):
                    sec = np.arctan2(rv[1], rv[0])
                    sec = sec * 180 / np.pi
                    if cs[1,2]<0:
                        sec *= -1
                    sec -= 90
                    #if prim < -90:
                    #    sec += 180
                else:
                    sec = np.arctan2(rv[2], rv[0])
                    sec = sec * 180 / np.pi - 90
                    #if prim > 90:
                    #    sec += 90
                    #elif prim < -90:
                    #    sec += 90
                    #else:
                    #    sec -= 90
                    #if cs[1,1]<0 and cs[2,2]<0:
                    #    sec -= 180

                prims.append(prim)
                secs.append(sec)


                ims.append(ds.pixel_array)
                #gammas.append(np.array(ds[0x0021,0x1028][0][0x0021,0x1042].value))
                window_center.append(float(ds.WindowCenter))
                window_width.append(float(ds.WindowWidth))
                water_value.append(float(ds[0x0021,0x1049].value))
                if ds[0x021,0x1010].VR == "US":
                    gamma_in.append(float(ds[0x021,0x1010].value))
                else:
                    gamma_in.append(float(int.from_bytes(ds[0x021,0x1010].value, "little", signed=False)))
                if ds[0x021,0x1011].VR == "US":
                    gamma_out.append(float(ds[0x021,0x1011].value))
                else:
                    gamma_out.append(float(int.from_bytes(ds[0x021,0x1011].value, "little", signed=False)))
                if ds[0x0019,0x1008].VR == "US":
                    percent_gain.append(ds[0x0019,0x1008].value)
                else:
                    percent_gain.append(float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False)))

    print("create numpy arrays")
    kvs = np.array(kvs)
    mas = np.array(mas)
    μas = np.array(μas)
    #gammas = np.array(gammas)
    gamma_in = np.array(gamma_in)
    gamma_out = np.array(gamma_out)
    percent_gain = np.array(percent_gain)
    ts = np.array(ts)
    thetas = np.array(thetas)
    phis = np.array(phis)
    ims = np.array(ims)
    window_center = np.array(window_center)
    window_width = np.array(window_width)
    #water_value = np.array(water_value)
    coord_systems = np.array(coord_systems)
    sids = np.array(sids)
    sods = np.array(sods)

    cs = coord_systems
    rots = np.arccos((cs[:,0,0]+cs[:,1,1]+cs[:,2,2]-1)/2)
    e1s = (cs[:,2,1]-cs[:,1,2]) / (2*np.sin(rots))
    e2s = (cs[:,0,1]-cs[:,2,1]) / (2*np.sin(rots))
    e3s = (cs[:,1,0]-cs[:,0,1]) / (2*np.sin(rots))
    ls = np.sqrt(e1s**2 + e2s**2 + e3s**2)
    es = np.vstack((e1s/ls, e2s/ls, e3s/ls)).T
    
    rv = cs[:,:,2]
    rv /= np.vstack((np.sum(rv, axis=-1),np.sum(rv, axis=-1),np.sum(rv, axis=-1))).T
    prim = np.arctan2(np.sqrt(rv[:,0]**2+rv[:,1]**2), rv[:,2])
    prim = np.arccos(rv[:,2] / np.sqrt(rv[:,0]**2+rv[:,1]**2+rv[:,2]**2) )
    sec = 0.5*np.pi-np.arctan2(rv[:,1], rv[:,0])
    prim = prim*180 / np.pi
    sec = sec * 180 / np.pi

    prim[cs[:,1,2]<0] *= -1
    sec[cs[:,1,2]<0] -= 180
    #sec[cs[:,1,2]<0] *= -1
    prim[np.bitwise_and(cs[:,2,2]<0, cs[:,1,1]>0) ] -= 180
    prim[cs[:,1,0]>0] -= 180

    thetas = prim
    phis = sec

    #print(prims)
    #print(secs)
    #print(coord_systems[170:180, :, 0])
    #print(coord_systems[170:180,:, 1])
    #print(coord_systems[170:180,:, 2])
    #print(coord_systems[170:180,:, 3])

    #plt.plot(np.arange(len(thetas)), prims)
    #plt.plot(np.arange(len(thetas)), thetas)
    #plt.show()
    #plt.figure()
    #plt.plot(thetas, secs)
    #plt.plot(thetas, phis)
    #plt.figure()
    #plt.plot(prims, secs)
    #plt.plot(prims, phis)
    #plt.show()
    #exit(0)

    dicom_angles = np.vstack((thetas*np.pi/180.0, phis*np.pi/180.0, np.zeros_like(thetas))).T
    if reg_angles:
        angles = read_reg_angles(prefix, dicom_angles)
        #diff = np.abs(dicom_angles[:,0]-angles[:,0])
        #print(dicom_angles[diff>0.1,0], angles[diff>0.1,0], diff[diff>0.1])
        #diff = np.abs(dicom_angles[:,1]-angles[:,1])
        #print(dicom_angles[diff>0.1,1], angles[diff>0.1,1], diff[diff>0.1])
        #diff = np.abs(dicom_angles[:,2]-angles[:,2])
        #print(dicom_angles[diff>0.1,2], angles[diff>0.1,2], diff[diff>0.1])
    else:
        angles = dicom_angles
        
    if False:
        #filt = filter_images(ims, ts, angles, mas)
        #filt = np.ones((ims.shape[0],), dtype=bool)
        filt = np.linalg.norm(coord_systems[:,:,2], axis=-1)<1.1
        ims = ims[filt]
        μas = μas[filt]
        mas = mas[filt]
        kvs = kvs[filt]
        #gammas = gammas[filt]
        gamma_in = gamma_in[filt]
        gamma_out = gamma_out[filt]
        percent_gain = percent_gain[filt]
        angles = angles[filt]
        window_center = window_center[filt]
        window_width = window_width[filt]
        #water_value = water_value[filt]
        coord_systems = coord_systems[filt]
        sids = sids[filt]
        sods = sods[filt]
        #print(percent_gain)


    #diff = np.linspace(angles[0,0],angles[-1,0],len(angles))-angles[:,0]
    #plt.plot(np.arange(len(diff)), diff)
    #plt.show()

    ims, i0s = normalize(ims, μas, kvs, None, window_center, window_width, percent_gain)

    return ims, i0s, angles, coord_systems, sids, sods

def read_reg_angles(prefix, dicom_angles):
    name = 'vectors_' + prefix.split('_', maxsplit=1)[1][:-1] + '.mat'
    #print(name)
    if os.path.isfile(name):
        import scipy.io
        vecs = scipy.io.loadmat(name)['newVectors']
        angles = []
        for vec in vecs:
            phi = np.arccos(vec[2] / np.sqrt(vec[2]*vec[2]+vec[1]*vec[1]+vec[0]*vec[0])) - np.pi*0.5
            theta = np.arctan2(vec[0], vec[1]) - np.pi*0.5
            if theta>np.pi:
                theta -= 2*np.pi
            if theta<-np.pi:
                theta += 2*np.pi
            angles.append([theta, -phi, 0])
        return np.array(angles)
    return dicom_angles

def create_geo(ims_shape, size, spacing):
    
    geo = tigre.geometry(mode='cone', nVoxel=np.array([512,512,512]),default=True)
    geo.nDetector = np.array((ims_shape[1], ims_shape[2]))             # number of pixels              (px)
    geo.dDetector = np.array((0.154*1920/ims_shape[1], 0.154*2480/ims_shape[2]))             # size of each pixel            (mm)
    geo.sDetector = geo.dDetector * geo.nDetector

    dSD = 1198
    dSI = 785
    geo.DSD = dSD
    geo.DSO = dSI

    geo.nVoxel = np.roll(size+20, 1)           # number of voxels              (vx)
    geo.sVoxel = np.roll((size+20)*spacing, 1)    # total size of the image       (mm)
    geo.dVoxel = np.roll(spacing, 1)

    return geo

def reco(prefix, ims, angles, geo, origin, size, spacing):


    print("start backprojecon")
    proctime = time.perf_counter()
    image = tigre.Atb(ims,geo,angles)
    save_image(image, prefix+"reco_tigre_Atb.nrrd", origin, spacing, True)
    print("Runtime: ", time.perf_counter() - proctime)
    print("start fdk")
    proctime = time.perf_counter()
    image = tigre.algorithms.fdk(ims,geo,angles)
    save_image(image, prefix+"reco_tigre_fdk.nrrd", origin, spacing, True)
    print("Runtime: ", time.perf_counter() - proctime)
    #print("start ML_OSTR")
    #niter = 500
    #proctime = time.perf_counter()
    #image = mlem.ML_OSTR(ims,geo,angles,niter)
    #print(np.mean(image), np.min(image), np.max(image), np.median(image), image.dtype)
    #save_image(image, prefix+"reco_tigre_ml_ostr.nrrd", origin, spacing)
    #print("Runtime: ", time.perf_counter() - proctime)
    #print("start PL_OSTR")
    #niter = 100
    #proctime = time.perf_counter()
    #image = mlem.PL_OSTR(ims,geo,angles,niter)
    #save_image(image, prefix+"reco_tigre_pl_ostr.nrrd", origin, spacing)
    #print("Runtime: ", time.perf_counter() - proctime)
    #print("start CCA")
    #niter = 100
    #proctime = time.perf_counter()
    #image = mlem.CCA(ims,geo,angles,niter)
    #save_image(image, prefix+"reco_tigre_cca.nrrd", origin, spacing)
    #print("Runtime: ", time.perf_counter() - proctime)
    #return
    #print("start mlem0")
    #niter = 100
    #proctime = time.perf_counter()
    #image = mlem.mlem0(ims,geo,angles,niter)
    #save_image(image, prefix+"reco_tigre_mlem0.nrrd", origin, spacing)
    #print("Runtime: ", time.perf_counter() - proctime)
    #return
    #print("start ossart")
    #niter = 30
    #proctime = time.perf_counter()
    #image = tigre.algorithms.ossart(ims,geo,angles,niter,blocksize=20)
    #save_image(image, prefix+"reco_tigre_ossart.nrrd", origin, spacing)
    #print("Runtime: ", time.perf_counter() - proctime)
    print("start sirt")
    niter = 30
    proctime = time.perf_counter()
    image = tigre.algorithms.sirt(ims,geo,angles,niter,blocksize=20)
    save_image(image, prefix+"reco_tigre_sirt.nrrd", origin, spacing, True)
    print("Runtime: ", time.perf_counter() - proctime)
    print("start cgls")
    niter = 15
    proctime = time.perf_counter()
    image = tigre.algorithms.cgls(ims,geo,angles,niter)
    save_image(image, prefix+"reco_tigre_cgls.nrrd", origin, spacing, True)
    print("Runtime: ", time.perf_counter() - proctime)
    #print("start fista")
    #niter = 70
    #proctime = time.perf_counter()
    #fistaout = tigre.algorithms.fista(ims,geo,angles,niter,hyper=2.e4)
    #image = sitk.GetImageFromArray(fistaout)
    #image.SetOrigin(origin[0])
    #image.SetDirection(origin[1])
    #sitk.WriteImage(image, prefix+"reco_tigre_fista.nrrd", origin, spacing)
    #print("Runtime: ", time.perf_counter() - proctime)
    print("start asd pocs")
    niter = 10
    proctime = time.perf_counter()
    image = tigre.algorithms.asd_pocs(ims,geo,angles,niter)
    save_image(image, prefix+"reco_tigre_asd_pocs.nrrd", origin, spacing, True)
    print("Runtime: ", time.perf_counter() - proctime)


def WriteAstraImage(im, path, astra_spacing, astra_zoom, image_out_mult=100): 
    sim = sitk.GetImageFromArray(im*image_out_mult*np.mean(astra_zoom))
    sim.SetSpacing(astra_spacing[[1,2,0]])
    sitk.WriteImage(sim, path)

def reco_astra(proj_data, name, astra_iter, proj_geom, vol_geom, astra_spacing, astra_zoom, real_image, origin, spacing, astra_algo="SIRT3D_CUDA"):
    if astra_iter == 0: return
    perftime = time.perf_counter()
    #sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "astra_"+name+"_sino.nrrd"))
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict(astra_algo)
    cfg['ReconstructionDataId'] = rec_id
    proj_id = astra.data3d.create('-sino', proj_geom, proj_data)
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, astra_iter)
    rec1 = astra.data3d.get(rec_id)
    rec_mult = astra_spacing[0]*astra_spacing[1]*astra_spacing[2]
    rec_mult = 1
    #print(rec1.shape)
    rec = np.swapaxes(rec1, 0, 2)
    rec = np.swapaxes(rec, 1,2)
    rec = rec[::-1, ::-1]
    #print(rec.shape)
    #WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_"+name+"_reco.nrrd"), astra_spacing, astra_zoom)
    sitk.WriteImage(sitk.GetImageFromArray(rec1), os.path.join("recos", "astra_"+name+"_reco.nrrd"))
    save_image(rec*rec_mult, "reg_astra_"+name+"_reco.nrrd", origin, spacing)
    #WriteAstraImage(real_image-rec*rec_mult, os.path.join("recos", "astra_"+name+"_error.nrrd"), astra_spacing, astra_zoom)
    #save_image(real_image-rec*rec_mult, "reg_astra_"+name+"_error.nrrd", origin, spacing)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    print("Astra "+name+": ", time.perf_counter()-perftime, np.sum(np.abs(real_image-rec)), np.log(np.sum(np.abs(real_image-rec))) )
    return rec1


def circle_mask(size):
    _xx, yy, zz = np.mgrid[:size[0], :size[1], :size[2]]
    tube = (yy - size[1]/2)**2 + (zz - size[2]/2)**2
    #sitk.WriteImage(sitk.GetImageFromArray(tube), "test.nrrd")
    mask = tube > (size[1]/2-5)**2
    #sitk.WriteImage(sitk.GetImageFromArray(np.array(mask,dtype=int)), "mask.nrrd")
    return mask

def save_plot(data, prefix, title):
    data = np.array(data)
    plt.figure()
    plt.plot(np.array(list(range(len(data[:, 0])))), data[:, 1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("recos", prefix + "plot_" + title + ".png"))
    with open(os.path.join("recos", prefix + "plot_" + title + ".csv"), "w") as f:
        f.writelines([str(t)+";"+str(v)+"\n" for t,v in data])

def save_image(image, filename, origin, spacing, switch_axes=False, hu_transform=True, crop=True):
    perftime = time.perf_counter()
    image = np.array(image[:], dtype=float)
    if hu_transform:
        μW = 0.019286726
        μA = 0.000021063006
        image = np.array(image, dtype=np.float32)
        if crop:
            #print(image.shape)
            if (margin>0).all():
                #image = image[margin[0]+40:-margin[0]-40,margin[1]+55:-margin[1]-55,margin[2]+55:-margin[2]-55]
                image = image[margin[0]:-margin[0],margin[1]:-margin[1],margin[2]:-margin[2]]
                #image = scipy.ndimage.zoom(image, np.array([395, 512, 512])/image.shape)
        image = 1000.0*((image - μW)/(μW-μA))
        mask = circle_mask(image.shape)
        image[mask] = 0.0
    else:
        #image = image-np.min(image)
        #image[image>np.mean(image)+4*np.std(image)] = np.mean(image)+4*np.std(image)
        #image = image / np.max(image)
        #image = image*100 - 50
        #image[image>5] = 5
        image = image
        if crop:
            if (margin > 0).all():
                image = image[margin[0]:-margin[0],margin[1]:-margin[1],margin[2]:-margin[2]]
            #image = scipy.ndimage.zoom(image, np.array([395, 512, 512])/image.shape)
    #name = 'vectors_' + prefix.split('_', maxsplit=1)[1][:-1] + '.mat'
    if switch_axes:
        image = sitk.GetImageFromArray(np.swapaxes(image, 1,2)[::-1, :, ::-1])
    else:
        image = sitk.GetImageFromArray(image)
    if origin is not None:
        image.SetOrigin(origin[0])
        image.SetDirection([np.round(o) for o in origin[1]])
    if spacing is not None:
        image.SetSpacing(spacing)
    sitk.WriteImage(image, os.path.join("recos", filename))
    print("Saving image: ", filename, " took: ", time.perf_counter()-perftime, " s")

def read_cbct_info(path):

    # open 2 fds
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    # save the current file descriptors to a tuple
    save = os.dup(1), os.dup(2)
    # put /dev/null fds on 1 and 2
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)

    # *** run the function ***
    

    sitk.ProcessObject_GlobalDefaultDebugOff()
    sitk.ProcessObject_GlobalWarningDisplayOff()
    reader = sitk.ImageSeriesReader()
    reader.DebugOff()
    reader.GlobalWarningDisplayOff()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames([os.path.join(path, f) for f in sorted(os.listdir(path))])
    image = reader.Execute()
    

    # restore file descriptors so I can print the results
    os.dup2(save[0], 1)
    os.dup2(save[1], 2)
    # close the temporary fds
    os.close(null_fds[0])
    os.close(null_fds[1])

    size = np.array(image.GetSize())
    origin = image.GetOrigin()
    direction = image.GetDirection()
    spacing = np.array(image.GetSpacing())
    return (origin, direction), size, spacing, image

def save_plot(data, prefix, title):
    data = np.array(data)
    plt.figure()
    plt.plot(np.array(list(range(len(data[:, 0])))), data[:, 1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("recos", prefix + title + "_plot.png"))
    with open(os.path.join("recos", prefix + title + "_plot.csv"), "w") as f:
        f.writelines([str(t)+";"+str(v)+"\n" for t,v in data])
    plt.close()

#margin = np.array([30, 85, 85])
margin = np.array([0,0,0])

def calc_pos_from_axis(axis_angles):
    G = 1300 #1100 1300 1500
    a0_a1 = np.array([0, 0, 750])
    a1_a2 = np.array([0, 350, 0])
    a2_a3 = np.array([0, 0, 1250])
    a3_a3 = np.array([0, 0 ,-55])
    a3_a4 = np.array([0, G, 0])
    a4_a5 = np.array([0, 0, 230])

    pos = a0_a1 + axis_angles[0]*a1_a2 + axis_angles[1]*a2_a3 + a3_a3 + axis_angles[2]*a3_a4 + axis_angles[4]*a4_a5

def main():
    data = [
    #('201020_imbu_cbct_', '.\\output\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV'),
    #('201020_imbu_sin_', '.\\output\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD'),
    #('201020_imbu_opti_', '.\\output\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD'),
    #('201020_imbu_circ_', '.\\output\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD'),
    #('201020_noimbu_cbct_', '.\\output\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV'),
    #('201020_noimbu_opti_', '.\\output\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD'),
    #('210111_balt_cbct_', '.\\output\\CKM4Baltimore\\CBCT_SINO')
    ('201207_cbct', '.\\output\\CKM\\CBCT\\20201207-093148.064000-20sDCT Head 70kV'),
    ('201207_circ', '.\\output\\CKM\\Circ Tomo 2. Versuch\\20201207-105441.287000-P16_DR_HD'),
    ('201207_eight', '.\\output\\CKM\\Eight die Zweite\\20201207-143732.946000-P16_DR_HD'),
    ('201207_opti', '.\\output\\CKM\\Opti Traj\\20201207-163001.022000-P16_DR_HD'),
    ('201207_sin', '.\\output\\CKM\\Sin Traj\\20201207-131203.754000-P16_Card_HD'),
    ('201207_tomo', '.\\output\\CKM\\Tomo\\20201208-110616.312000-P16_DR_HD'),
    ]

    origin, size, spacing, image = read_cbct_info(r".\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
    #origin, size, spacing, image = read_cbct_info(r".\output\CKM4Baltimore\CBCT_2021_01_11_16_04_12")
    #origin, size, spacing, image = read_cbct_info(r".\output\CKM4Baltimore\CBCT_2021_01_11_11_55_50")
    
    

    for prefix, path in data:
        print(prefix, path)
        proctime = time.perf_counter()
        try:
            ims, i0s, angles, coord_systems, sids, sods = read_dicoms(path, prefix, reg_angles=False)

            angles_one = np.ones_like(angles[:,0])
            geo = create_geo(ims.shape, size-20, spacing)
            
            out_shape = geo.nVoxel
            detector_shape = np.array((1920,2480))
            detector_mult = np.floor(detector_shape / np.array(ims.shape[1:]))
            detector_shape = np.array(ims.shape[1:])
            detector_spacing = np.array((0.125, 0.125)) * detector_mult
            dSD = 1198
            dSI = 785
            angles_astra=angles
            angles_astra[:,1] += angles_one*0.5*np.pi
            angles_astra[:,2] += angles_one*np.pi
            #print(angles.shape, angles_astra.shape)
            name = prefix
            real_image = sitk.GetArrayFromImage(image)
            margin1 = np.array([20, 64, 64])
            image_zoom = (np.array(real_image.shape) + 2*margin1)/real_image.shape
            print("zoom image")
            cube_astra = np.zeros((np.array(real_image.shape) + 2*margin))
            #cube_astra = scipy.ndimage.zoom(real_image, image_zoom, order=2)
            #cube_astra = real_image
            #if (margin>0).all():
            #    cube_astra = np.zeros(np.array(zoomed_image.shape)+2*margin)
            #    cube_astra[margin[0]:-margin[0],margin[1]:-margin[1],margin[2]:-margin[2]] = zoomed_image
            #else:
            #    cube_astra = zoomed_image
            #cube_astra = zoomed_image
            astra_spacing = np.array(spacing) / image_zoom
            #print(real_image.shape, cube_astra.shape, spacing)
            #print(astra_spacing, image_zoom, 1/np.min(astra_spacing), 1/np.min(spacing), 1.5/np.min(spacing))

            if coord_systems is None:
                proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dSI, dSD-dSI, 1/np.mean(astra_spacing))
            else:
                proj_geom_v, prims, filt = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods, sids-sods, 1/np.min(astra_spacing))
                #proj_geom_c = astra.create_proj_geom('cone', detector_spacing[0]/np.mean(astra_spacing), detector_spacing[1]/np.mean(astra_spacing), detector_shape[0], detector_shape[1], prims, dSI/np.mean(astra_spacing), (dSD-dSI)/np.mean(astra_spacing))


            if False:
                plt.figure()
                plt.title("angles")
                plt.plot(np.arange(len(prims)), prims)
                plt.figure()
                plt.title("x")
                plt.plot(np.arange(len(proj_geom_v['Vectors'])), proj_geom_v["Vectors"][:,0])
                #proj_geom_v["Vectors"][:,0] = scipy.signal.savgol_filter(proj_geom_v["Vectors"][:,0], 2*len(proj_geom_v['Vectors'])//2-1, 2)
                proj_geom_v["Vectors"][:,0] = scipy.signal.savgol_filter(proj_geom_v["Vectors"][:,0], 51, 1)
                #proj_geom_v["Vectors"][:,0] = np.zeros_like(proj_geom_v["Vectors"][:,0])
                plt.plot(np.arange(len(proj_geom_v['Vectors'])), proj_geom_v["Vectors"][:,0])
                plt.figure()
                plt.title("y")
                plt.plot(np.arange(len(proj_geom_v['Vectors'])), proj_geom_v["Vectors"][:,1])
                plt.figure()
                plt.title("z")
                plt.plot(np.arange(len(proj_geom_v['Vectors'])), proj_geom_v["Vectors"][:,2])
                plt.show()
                #print(proj_geom_v1["Vectors"][0])
                #print(proj_geom_v["Vectors"][0])
                

            #print(np.array(cube_astra.shape)[[1,2,0]])
            #print(np.array(cube_astra.shape)[[1,0,2]])
            #vol_geom = astra.create_vol_geom(cube_astra.shape[1], cube_astra.shape[2], cube_astra.shape[0])
            vol_geom = astra.create_vol_geom(cube_astra.shape[1], cube_astra.shape[0], cube_astra.shape[2])

            ims_astra = np.swapaxes(ims, 0, 1)
            i0s_astra = np.ones_like(ims_astra)
            for i in range(i0s_astra.shape[1]):
                i0s_astra[:,i,:] *= i0s[i]

            #print(ims_astra.shape, cube_astra.shape, detector_shape, detector_spacing)
            #print(ims_astra.dtype, i0s.dtype, cube_astra.dtype)
            #print(ims.shape, ims_astra.shape, detector_shape, i0s.shape, i0s_astra.shape)
            print("start reco")
            sitk.WriteImage(sitk.GetImageFromArray(ims_astra), "recos/astra_"+name+"_input.nrrd")
            sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(np.swapaxes(real_image,0,2), 0, 1)[::-1,:,::-1]), "recos/astra_"+name+"_input_image.nrrd")

            volume_id = astra.data3d.create('-vol', vol_geom, np.swapaxes(np.swapaxes(real_image,0,2), 0, 1)[::-1,:,::-1])
            proj_id = astra.data3d.create('-sino', proj_geom_v, 0)
            algString = 'FP3D_CUDA'
            cfg = astra.astra_dict(algString)
            cfg['ProjectionDataId'] = proj_id
            cfg['VolumeDataId'] = volume_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(volume_id)
            proj_data = np.zeros_like(ims_astra)
            proj_data[:,filt] = astra.data3d.get(proj_id)
            sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "astra_"+name+"_sino.nrrd"))
            astra.data3d.delete(proj_id)


            rec = reco_astra(ims_astra[:,filt], name+"fdk_", 1, proj_geom_v, vol_geom, astra_spacing, image_zoom, cube_astra, origin, spacing, astra_algo="FDK_CUDA")
            
            volume_id = astra.data3d.create('-vol', vol_geom, rec)
            proj_id = astra.data3d.create('-sino', proj_geom_v, 0)
            algString = 'FP3D_CUDA'
            cfg = astra.astra_dict(algString)
            cfg['ProjectionDataId'] = proj_id
            cfg['VolumeDataId'] = volume_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(volume_id)
            proj_data = np.zeros_like(ims_astra)
            proj_data[:,filt] = astra.data3d.get(proj_id)
            astra.data3d.delete(proj_id)
            sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "astra_"+name+"_sino_fdk.nrrd"))

            #print(np.size(filt), np.count_nonzero(filt))

            #exit(0)
            rec = reco_astra(ims_astra[:,filt], name+"sirt_", 50, proj_geom_v, vol_geom, astra_spacing, image_zoom, cube_astra, origin, spacing, astra_algo="SIRT3D_CUDA")

            volume_id = astra.data3d.create('-vol', vol_geom, rec)
            proj_id = astra.data3d.create('-sino', proj_geom_v, 0)
            algString = 'FP3D_CUDA'
            cfg = astra.astra_dict(algString)
            cfg['ProjectionDataId'] = proj_id
            cfg['VolumeDataId'] = volume_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(volume_id)
            proj_data = astra.data3d.get(proj_id)
            astra.data3d.delete(proj_id)
            sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "astra_"+name+"_sino_sirt.nrrd"))
            #exit(0)
            #cube_astra = np.swapaxes(cube_astra, 1, 2)
            #rec = np.swapaxes(rec, 0, 2)
            image_shape = rec.shape
            initial = np.zeros(image_shape)
            initial = np.array(rec)
            stat_iter = 50
            e = 2
            e2 = 5
            for e2 in [4]:#,5,6]:
                b = 10**(e)
                β = 10**(e2/2)
                for p in [1,2,"quad_wls", "huber_wls"]:
                    name1 = "pl_pcg_qs_ls"+str(e)+"-"+str(p)+"_"+str(e2)
                    rec_mult = astra_spacing[0]*astra_spacing[1]*astra_spacing[2]
                    rec_mult = 1
                    if p in ["quad_wls", "huber_wls"]:
                        rec_algo = mlem.pl_pcg_qs_ls(ims_astra[:,filt], image_shape, proj_geom_v, angles_astra, stat_iter, initial=initial, real_image=np.swapaxes(np.swapaxes(cube_astra,0,2), 0, 1)[::-1,::-1], b=i0s_astra[:,filt], β=β, p=p)
                    else:
                        rec_algo = mlem.pl_iot(ims_astra[:,filt], image_shape, proj_geom_v, angles_astra, stat_iter, initial=initial, real_image=np.swapaxes(np.swapaxes(cube_astra,0,2), 0, 1)[::-1,::-1], b=i0s_astra[:,filt], β=β, p=p)
                    perftime = time.perf_counter()
                    for i, rec in enumerate(rec_algo):
                        if type(rec) is list:
                            save_plot(rec, name1, name1)
                        elif type(rec) is tuple:
                            save_plot(rec[0], name1 +"_error_", name)
                            #save_plot(rec[1], name+"_obj_func_", name)
                            #WriteAstraImage(rec[2]*rec_mult, os.path.join("recos", name+"_reco.nrrd"), astra_spacing, image_zoom)
                            rec1 = np.swapaxes(rec[2], 0, 2)
                            rec1 = np.swapaxes(rec1, 1,2)
                            rec1 = rec1[::-1, ::-1]
                            save_image(rec1*rec_mult, "reg_"+prefix+"_"+name1+"_reco.nrrd", origin, spacing)
                            #WriteAstraImage(cube_astra-rec[2], os.path.join("recos", name+"_error.nrrd"), astra_spacing, image_zoom)
                            #save_image(cube_astra-rec*rec_mult, "reg_"+prefix+"_"+name+"_error.nrrd", origin, spacing)
                        else:
                            #sitk.WriteImage(sitk.GetImageFromArray(rec*image_out_mult), os.path.join("recos", name+"_" + str(i) + "_reco.nrrd"))
                            #sitk.WriteImage(sitk.GetImageFromArray((cube-rec)*image_out_mult), os.path.join("recos", name+"_"+str(i)+"_error.nrrd"))
                            #sitk.WriteImage(sitk.GetImageFromArray(rec*image_out_mult), os.path.join("recos", name+"_reco.nrrd"))
                            #sitk.WriteImage(sitk.GetImageFromArray((cube_astra-rec)*image_out_mult), os.path.join("recos", name+"_error.nrrd"))
                            #WriteAstraImage(rec*rec_mult, os.path.join("recos", name+"_reco_" + str(i) + ".nrrd"), astra_spacing, image_zoom)
                            rec1 = np.swapaxes(rec, 0, 2)
                            rec1 = np.swapaxes(rec1, 1,2)
                            rec1 = rec1[::-1, ::-1]
                            #save_image(rec*rec_mult, "reg_"+name+"_reco_" + str(i) + ".nrrd", origin, spacing)
                            #WriteAstraImage(cube_astra-rec, os.path.join("recos", name+"_error_" + str(i) + ".nrrd"), astra_spacing, image_zoom)
                            #save_image(cube_astra-rec*rec_mult, "reg_"+name+"_error_" + str(i) + ".nrrd", origin, spacing)
                            print(i, name1+": ", time.perf_counter()-perftime, np.sum(np.abs(cube_astra-rec1)), np.log(np.sum(np.abs(cube_astra-rec1))))

            volume_id = astra.data3d.create('-vol', vol_geom, rec[2])
            proj_id = astra.data3d.create('-sino', proj_geom_v, 0)
            algString = 'FP3D_CUDA'
            cfg = astra.astra_dict(algString)
            cfg['ProjectionDataId'] = proj_id
            cfg['VolumeDataId'] = volume_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(volume_id)
            proj_data = astra.data3d.get(proj_id)
            astra.data3d.delete(proj_id)
            sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "astra_"+name+"_sino_pl.nrrd"))

        except Exception as e:
            print(str(e))
            raise

        print("Runtime :", time.perf_counter() - proctime)

if __name__ == "__main__":
    main()