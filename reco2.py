import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import os.path
import tigre
import time
import i0

def conv_time(time):
    h = float(time[:2])*60*60
    m = float(time[2:4])*60
    s = float(time[4:])
    return h+m+s


def normalize(images, mAs_array, kV_array, gammas, window_center, window_width):
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
    kVs[70] = f

    fs = []
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

    #norm_images = images / fs
    norm_images = np.zeros((images.shape[0], images.shape[1]//4-20, images.shape[2]//4-20), dtype=np.float32)
    minWindow = window_center - window_width / 2
    windowScale = window_width / 4096
    maxValue = minWindow + window_width
    
    igamma = inverse_lut(gamma)
    for i in range(len(fs)):
        ilut = inverse_lut(gammas[i])
        norm_img = images[i,40:-40:4,40:-40:4]
        #norm_img[norm_img<0] = 0
        #norm_img[norm_img>4095] = 4095

        #norm_img = ilut[norm_img]
        #norm_img = gamma[norm_img]

        #norm_img = (minWindow[i] + windowScale[i]*norm_img)
        #initial = fs[i]
        #fs[i] = gammas[i][igamma[int(initial)]]
        norm_images[i] = -np.log(norm_img / fs[i])
    print(np.mean(fs), np.median(fs), np.max(fs), np.min(fs))
    print(np.mean(mAs_array), np.median(mAs_array), np.max(mAs_array), np.min(mAs_array))
    print(np.mean(images), np.median(images), np.max(images), np.min(images))
    print(np.mean(norm_images), np.median(norm_images), np.max(norm_images), np.min(norm_images))
    return norm_images

def filter_images(ims, ts, angles, mas):
    print("filter images")  
    filt = np.zeros_like(ts, dtype=bool)

    for i, (angle, ma) in enumerate(zip(angles, mas)):
        if ma <= np.min(mas[np.bitwise_and(angles[:,0]==angle[0], angles[:,1]==angle[1])]):
            filt[i] = True
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

def read_dicoms(indir):
    print("read dicoms")
    kvs = []
    mas = []
    μas = []
    ts = []
    thetas = []
    phis = []
    ims = []
    gammas = []
    window_center = []
    window_width = []
    #sid = []
    for root, dirs, files in os.walk(indir):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            #read DICOM files
            ds = pydicom.dcmread(path)

            if "PositionerPrimaryAngleIncrement" in dir(ds):
                #ts.append(conv_time(ds.AcquisitionTime))
                #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
                ims = ds.pixel_array
                if (ds.BitsStored == 16):
                    ims = np.array(ims*(2.0**14/2.0**16), dtype=int)
                ts = list(range(len(ims)))
                thetas = ds.PositionerPrimaryAngleIncrement
                phis = ds.PositionerSecondaryAngleIncrement
                window_center = [float(ds.WindowCenter)] * len(ts)
                window_width = [float(ds.WindowCenter)] * len(ts)
                gammas = [np.array(ds[0x0021,0x1028][0][0x0021,0x1042].value)] * len(ts)
                #sid = ds[0x0021,0x1031].value
                xray_info = np.array(ds[0x0021,0x100F].value)
                kvs = xray_info[0::4]
                mas = xray_info[1::4]*xray_info[2::4]*0.001
                μas = xray_info[2::4]*0.001

                thetas2 = np.array(ds[0x0021,0x1068].value)
                phis2 = np.array(ds[0x0021,0x1072].value)/100

                #print(thetas-thetas2/100)
                #print(phis-phis2)
                #angulation = ds[0x0021,0x105D].value
                #orbital = ds[0x0021,0x105E].value
            else:
                #ts.append(conv_time(ds.AcquisitionTime))
                #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
                ts.append(len(ts))
                kvs.append(float(ds.KVP))
                mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                μas.append(float(ds[0x0021,0x1004].value)*0.001)
                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))
                ims.append(ds.pixel_array)
                gammas.append(np.array(ds[0x0021,0x1028][0][0x0021,0x1042].value))
                window_center.append(float(ds.WindowCenter))
                window_width.append(float(ds.WindowCenter))

    print("create numpy arrays")
    kvs = np.array(kvs)
    mas = np.array(mas)
    μas = np.array(μas)
    gammas = np.array(gammas)
    ts = np.array(ts)
    thetas = np.array(thetas)
    phis = np.array(phis)
    ims = np.array(ims)
    window_center = np.array(window_center)
    window_width = np.array(window_width)

    dicom_angles = np.vstack((thetas*np.pi/180.0, phis*np.pi/180.0, np.zeros_like(thetas))).T
    angles = read_reg_angles(prefix, dicom_angles)

    diff = np.abs(dicom_angles[:,0]-angles[:,0])
    print(dicom_angles[diff>0.1,0], angles[diff>0.1,0], diff[diff>0.1])

    diff = np.abs(dicom_angles[:,1]-angles[:,1])
    print(dicom_angles[diff>0.1,1], angles[diff>0.1,1], diff[diff>0.1])

    diff = np.abs(dicom_angles[:,2]-angles[:,2])
    print(dicom_angles[diff>0.1,2], angles[diff>0.1,2], diff[diff>0.1])

    filt = filter_images(ims, ts, angles, mas)
    ims = ims[filt]
    μas = μas[filt]
    mas = mas[filt]
    kvs = kvs[filt]
    gammas = gammas[filt]
    angles = angles[filt]
    window_center = window_center[filt]
    window_width = window_width[filt]

    ims = normalize(ims, μas, kvs, gammas, window_center, window_width)

    return ims, angles

def read_reg_angles(prefix, dicom_angles):
    name = 'vectors_' + prefix.split('_', maxsplit=1)[1][:-1] + '.mat'
    print(name)
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

def reco(prefix, path, origin, size, spacing):
    ims, angles = read_dicoms(path)

    geo = tigre.geometry(mode='cone', nVoxel=np.array([512,512,512]),default=True)
    geo.nDetector = np.array((ims.shape[1], ims.shape[2]))             # number of pixels              (px)
    geo.dDetector = np.array((0.154*1920/ims.shape[1], 0.154*2480/ims.shape[2]))             # size of each pixel            (mm)
    geo.sDetector = geo.dDetector * geo.nDetector

    dSD = 1198
    dSI = 785
    geo.DSD = dSD
    geo.DSO = dSI

    geo.nVoxel = size           # number of voxels              (vx)
    geo.sVoxel = size*spacing    # total size of the image       (mm)
    geo.dVoxel = spacing

    print("start fdk")
    proctime = time.process_time()
    image = tigre.algorithms.fdk(ims,geo,angles)
    save_image(image, prefix+"reco_tigre_fdk.nrrd")
    print("Runtime: ", time.process_time() - proctime)
    print("start ossart")
    niter = 30
    proctime = time.process_time()
    image = tigre.algorithms.ossart(ims,geo,angles,niter,blocksize=20)
    save_image(image, prefix+"reco_tigre_ossart.nrrd")
    print("Runtime: ", time.process_time() - proctime)
    print("start cgls")
    niter = 15
    proctime = time.process_time()
    image = tigre.algorithms.cgls(ims,geo,angles,niter)
    save_image(image, prefix+"reco_tigre_cgls.nrrd")
    print("Runtime: ", time.process_time() - proctime)
    #print("start fista")
    #niter = 70
    #proctime = time.process_time()
    #fistaout = tigre.algorithms.fista(ims,geo,angles,niter,hyper=2.e4)
    #image = sitk.GetImageFromArray(fistaout)
    #image.SetOrigin(origin[0])
    #image.SetDirection(origin[1])
    #sitk.WriteImage(image, prefix+"reco_tigre_fista.nrrd")
    #print("Runtime: ", time.process_time() - proctime)
    print("start asd pocs")
    niter = 10
    proctime = time.process_time()
    image = tigre.algorithms.asd_pocs(ims,geo,angles,niter)
    save_image(image, prefix+"reco_tigre_asd_pocs.nrrd")
    print("Runtime: ", time.process_time() - proctime)

def circle_mask(size):
    _xx, yy, zz = np.mgrid[:size[0], :size[1], :size[2]]
    tube = (yy - size[1]/2)**2 + (zz - size[2]/2)**2
    #sitk.WriteImage(sitk.GetImageFromArray(tube), "test.nrrd")
    mask = tube > (size[1]/2 - 20)**2
    #sitk.WriteImage(sitk.GetImageFromArray(np.array(mask,dtype=int)), "mask.nrrd")
    return mask

def save_image(image, filename):
    μW = 0.019286726
    μA = 0.000021063006
    mask = circle_mask(image.shape)
    image[mask] = 0
    image = image[20:-20,20:-20,20:-20]
    image = 1000.0*((image - μW)/(μW-μA))
    name = 'vectors_' + prefix.split('_', maxsplit=1)[1][:-1] + '.mat'
    if not os.path.isfile(name):
        image = sitk.GetImageFromArray(image)
    else:
        image = sitk.GetImageFromArray(np.swapaxes(image, 1,2)[::-1,::-1])
    image.SetOrigin(origin[0])
    image.SetDirection(origin[1])
    image.SetSpacing(spacing)
    sitk.WriteImage(image, os.path.join("recos", filename))

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
    return (origin, direction), size, spacing

if __name__ == "__main__":
    data = [
    #('lumb_short_cbct_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-163837.330000\\5sDR Body'),
    #('lumb_cbct_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-163837.330000\\5sDR Body'),
    #('lumb_sin_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-123239.316000\\P16_DR_LD'),
    #('lumb_opti_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-150514.166000\\P16_DR_LD'),
    #('lumb_imb_cbct_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-093446.875000\\20sDCT Head 70kV'),
    #('lumb_imb_sin_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-122515.399000\\P16_DR_LD'),
    #('lumb_imb_opti_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-093446.875000\\P16_DR_LD'),
    #('lumb_imb_circ_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-140352.179000\\P16_DR_LD'),
    #('loc_imbu_cbct_', '.\\output\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV'),
    #('loc_imbu_sin_', '.\\output\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD'),
    #('loc_imbu_opti_', '.\\output\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD'),
    ('loc_imbu_circ_', '.\\output\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD'),
    #('loc_noimbu_cbct_', '.\\output\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV'),
    #('loc_noimbu_opti_', '.\\output\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD'),
    ]

    origin, size, spacing = read_cbct_info(r".\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")

    for prefix, path in data:
        print(prefix, path)
        proctime = time.process_time()
        try:
            reco(prefix, path, origin, size, spacing)
        except Exception as e:
            print(str(e))
            raise

        print("Runtime :", time.process_time() - proctime)