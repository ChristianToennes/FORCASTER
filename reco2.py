import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import os.path
import tigre
import time

def conv_time(time):
    h = float(time[:2])*60*60
    m = float(time[2:4])*60
    s = float(time[4:])
    return h+m+s


def normalize(images, mAs_array, kV_array, gammas, window_center, window_width):
    print("normalize images")
    kVs = np.array([40, 50, 60, 70, 80, 90, 100, 109, 120, 125])
    a = np.array([20.4125, 61.6163, 138.4021, 250.8008, 398.963, 586.5949, 794.5124, 1006.1, 1252.2, 1404.2202])
    b = np.array([677.964, 686.4824, 684.1844, 691.9573, 701.1038, 711.416, 729.8813, 750.0054, 791.9865, 796.101])
    kVs = np.array([70])
    a = np.array([98.2053])
    b = np.array([232.7655])

    fs = []
    for mAs, kV in zip(mAs_array, kV_array):
        if kV < kVs[0]:
            f = a[0]*mAs + b[0]
        elif kV > kVs[-1]:
            f = a[-1]*mAs + b[-1]
        elif kV in kVs:
            f = a[kVs==kV]*mAs + b[kVs==kV]
        else:
            i1, i2 = np.argsort(np.abs(kVs-kV))[:2]
            f1 = a[i1]*mAs + b[i1]
            f2 = a[i2]*mAs + b[i2]
            d1 = np.abs(kVs[i1]-kV)*1.0
            d2 = np.abs(kVs[i2]-kV)*1.0
            f = f1*(1.0-(d1/(d1+d2))) + f2*(1.0-(d2/(d1+d2)))
        
        #print(mAs, kV, f)
        fs.append(f)
    
    fs = np.array(fs).flatten()
    
    #norm_images = images / fs
    norm_images = np.zeros((images.shape[0], images.shape[1]-20, images.shape[2]-20), dtype=np.float32)
    minWindow = window_center - window_width / 2
    windowScale = window_width / 4096
    maxValue = minWindow + window_width
    for i in range(len(fs)):
        images[i] = (minWindow[i] + windowScale[i]*images[i])
        norm_images[i] = -np.log(images[i,10:-10,10:-10] / fs[i])
    print(np.mean(fs), np.median(fs), np.max(fs), np.min(fs))
    print(np.mean(mAs_array), np.median(mAs_array), np.max(mAs_array), np.min(mAs_array))
    print(np.mean(images), np.median(images), np.max(images), np.min(images))
    print(np.mean(norm_images), np.median(norm_images), np.max(norm_images), np.min(norm_images))
    return norm_images

def filter_images(ims, ts, angles, mas):
    print("filter images")  
    if len(ts) == 1:
        filt = 0
    else:
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
    for root, dirs, files in os.walk(indir):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            #read DICOM files
            ds = pydicom.dcmread(path)

            kvs.append(float(ds.KVP))
            mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime))
            μas.append(float(ds[0x0021,0x1004].value))
            if "PositionerPrimaryAngleIncrement" in dir(ds):
                #ts.append(conv_time(ds.AcquisitionTime))
                #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
                ts.append(len(ts))
                thetas.append(ds.PositionerPrimaryAngleIncrement)
                phis.append(ds.PositionerSecondaryAngleIncrement)
                ims.append(ds.pixel_array)
                window_center.append(float(ds.WindowCenter))
                window_width.append(float(ds.WindowCenter))
            else:
                #ts.append(conv_time(ds.AcquisitionTime))
                #ts.append(ds.AcquisitionTime + " - " + str(len(ts)))
                ts.append(len(ts))
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
    ims = ims[filt, ::4, ::4]
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
    image = ((1000.0*image)/(μW-μA)) - 1000.0
    image = sitk.GetImageFromArray(np.swapaxes(image, 1,2)[::-1,::-1])
    image.SetOrigin(origin[0])
    image.SetDirection(origin[1])
    image.SetSpacing(spacing)
    sitk.WriteImage(image, os.path.join("recos", filename))

def read_cbct_info(path):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([os.path.join(path, f) for f in sorted(os.listdir(path))])
    image = reader.Execute()
    size = np.array(image.GetSize())
    origin = image.GetOrigin()
    direction = image.GetDirection()
    spacing = np.array(image.GetSpacing())
    return (origin, direction), size, spacing

if __name__ == "__main__":
    data = [#('lumb_short_cbct_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-163837.330000\\5sDR Body'),
    #('lumb_cbct_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-163837.330000\\5sDR Body'),
    #('lumb_sin_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-123239.316000\\P16_DR_LD'),
    #('lumb_opti_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine\\20201013-150514.166000\\P16_DR_LD'),
    #('lumb_imb_cbct_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-093446.875000\\20sDCT Head 70kV'),
    #('lumb_imb_sin_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-122515.399000\\P16_DR_LD'),
    #('lumb_imb_opti_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-093446.875000\\P16_DR_LD'),
    #('lumb_imb_circ_', 'C:\\Users\\ich\\Source\\reco\\CKM_LumbalSpine_Imbu\\20201020-140352.179000\\P16_DR_LD'),
    ('loc_imbu_cbct_', 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV'),
    ('loc_imbu_sin_', 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD'),
    ('loc_imbu_opti_', 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD'),
    ('loc_imbu_circ_', 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD'),
    ('loc_noimbu_cbct_', 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV'),
    ('loc_noimbu_opti_', 'D:\\lumbal_spine_13.10.2020\\output\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD'),]

    origin, size, spacing = read_cbct_info(r"D:\lumbal_spine_13.10.2020\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")

    for prefix, path in data:
        print(prefix, path)
        proctime = time.process_time()
        try:
            reco(prefix, path, origin, size, spacing)
        except Exception as e:
            print(str(e))

        print("Runtime :", time.process_time() - proctime)