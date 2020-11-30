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

def conv_time(time):
    h = float(time[:2])*60*60
    m = float(time[2:4])*60
    s = float(time[4:])
    return h+m+s


def normalize(images, mAs_array, kV_array, gammas, window_center, window_width, water_value):
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
    wv = []
    igamma = inverse_lut(gamma)
    for i in range(len(fs)):
        norm_img = images[i,40:-40:4,40:-40:4]

        norm_img = norm_img / fs[i]
        
        norm_img = -np.log(norm_img)

        norm_images[i] = norm_img
    print(np.mean(fs), np.median(fs), np.max(fs), np.min(fs))
    print(np.mean(mAs_array), np.median(mAs_array), np.max(mAs_array), np.min(mAs_array))
    print(np.mean(images), np.median(images), np.max(images), np.min(images))
    print(np.mean(norm_images), np.median(norm_images), np.max(norm_images), np.min(norm_images))
    return norm_images

def filter_images(ims, ts, angles, mas):
    print("filter images")  
    filt = np.zeros(ts.shape, dtype=bool)

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

def read_dicoms(indir, reg_angles=True):
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
    water_value = []

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
                if (ds.BitsStored == 16):
                    ims = np.array(ims*(2.0**14/2.0**16), dtype=int)
                ts = list(range(len(ims)))
                thetas = ds.PositionerPrimaryAngleIncrement
                phis = ds.PositionerSecondaryAngleIncrement
                window_center = [float(ds.WindowCenter)] * len(ts)
                window_width = [float(ds.WindowWidth)] * len(ts)
                gammas = [np.array(ds[0x0021,0x1028][0][0x0021,0x1042].value)] * len(ts)
                #sid = ds[0x0021,0x1031].value
                xray_info = np.array(ds[0x0021,0x100F].value)
                kvs = xray_info[0::4]
                mas = xray_info[1::4]*xray_info[2::4]*0.001
                μas = xray_info[2::4]*0.001

                thetas2 = np.array(ds[0x0021,0x1068].value)
                phis2 = np.array(ds[0x0021,0x1072].value)/100

                water_value = [float(ds[0x0021,0x1049].value)]*len(ts)

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
                ims.append(ds.pixel_array)
                gammas.append(np.array(ds[0x0021,0x1028][0][0x0021,0x1042].value))
                window_center.append(float(ds.WindowCenter))
                window_width.append(float(ds.WindowWidth))
                water_value.append(float(ds[0x0021,0x1049].value))

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
    water_value = np.array(water_value)

    dicom_angles = np.vstack((thetas*np.pi/180.0, phis*np.pi/180.0, np.zeros_like(thetas))).T
    if reg_angles:
        angles = read_reg_angles(prefix, dicom_angles)

        diff = np.abs(dicom_angles[:,0]-angles[:,0])
        print(dicom_angles[diff>0.1,0], angles[diff>0.1,0], diff[diff>0.1])

        diff = np.abs(dicom_angles[:,1]-angles[:,1])
        print(dicom_angles[diff>0.1,1], angles[diff>0.1,1], diff[diff>0.1])

        diff = np.abs(dicom_angles[:,2]-angles[:,2])
        print(dicom_angles[diff>0.1,2], angles[diff>0.1,2], diff[diff>0.1])

    else:
        angles = dicom_angles
        
    filt = filter_images(ims, ts, angles, mas)
    ims = ims[filt]
    μas = μas[filt]
    mas = mas[filt]
    kvs = kvs[filt]
    gammas = gammas[filt]
    angles = angles[filt]
    window_center = window_center[filt]
    window_width = window_width[filt]
    water_value = water_value[filt]

    ims = normalize(ims, μas, kvs, gammas, window_center, window_width, water_value)

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

def reco_astra(prefix, real_image, ims, angles, geo, origin, size, spacing):

    out_shape = np.roll(size, 1)
    print("proj:", np.mean(ims), np.median(ims), np.max(ims), np.min(ims))
    print("start backprojection")
    proctime = time.perf_counter()
    image = utils.Atb_astra(out_shape, geo)(ims, free_memory=True)
    print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_astra_Atb.nrrd", origin, spacing, False, False)
    print("start fdk")
    proctime = time.perf_counter()
    #image = utils.FDK_astra(out_shape, geo)(ims, free_memory=True)
    print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_astra_fdk.nrrd", origin, spacing, False, False)

    #initial = image
    initial = np.ones_like(image)*0.1
    if True:
        print("start PL_C")
        astra.clear()
        niter = 500
        proctime = time.perf_counter()
        for i,image in enumerate(mlem.PL_C(ims,out_shape,geo,angles,niter, initial, real_image, β=10)):
            if type(image) is list:
                save_plot(image, prefix, "pl_c")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_pl_c_"+str(i)+".nrrd", origin, spacing, False, False)
    if True:
        print("start ML_OSTR")
        astra.clear()
        niter = 500
        proctime = time.perf_counter()
        for i,image in enumerate(mlem.ML_OSTR(ims,out_shape,geo,angles,niter,initial, real_image)):
            if type(image) is list:
                save_plot(image, prefix, "ml_ostr")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_ml_ostr_"+str(i)+".nrrd", origin, spacing, False, False)
    if True:
        print("start PL_OSTR")
        astra.clear()
        niter = 500
        proctime = time.perf_counter()
        for i,image in enumerate(mlem.PL_OSTR(ims,out_shape,geo,angles,niter,initial, real_image, β=10**2)):
            if type(image) is list:
                save_plot(image, prefix, "pl_ostr")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_pl_ostr_"+str(i)+".nrrd", origin, spacing, False, False)
    if True:
        print("start CCA")
        astra.clear()
        niter = 500
        proctime = time.perf_counter()
        for i,image in enumerate(mlem.CCA(ims,out_shape,geo,angles,niter,initial, real_image, β=10**2)):
            if type(image) is list:
                save_plot(image, prefix, "cca")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_cca_"+str(i)+".nrrd", origin, spacing, False, False)
    if True:
        print("start PIPLE")
        astra.clear()
        niter = 500
        proctime = time.perf_counter()
        for i,image in enumerate(mlem.PIPLE(ims,out_shape,geo,angles,niter,initial, real_image, βp=10**2)):
            if type(image) is list:
                save_plot(image, prefix, "piple")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_piple_"+str(i)+".nrrd", origin, spacing, False, False)
    if True:
        print("start PL_PSCD")
        astra.clear()
        niter = 500
        proctime = time.perf_counter()
        for i,image in enumerate(mlem.PL_PSCD(ims,out_shape,geo,angles,niter,initial, real_image, β=10**2)):
            if type(image) is list:
                save_plot(image, prefix, "pl_pcsd")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_pl_pcsd_"+str(i)+".nrrd", origin, spacing, False, False)
    #print("start tilley2017")
    #niter = 15
    #proctime = time.perf_counter()
    #image = mlem.mlem2(ims,out_shape,geo,angles,niter,initial)
    #print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_tigre_tilley.nrrd", origin, spacing, False, False)
    #print("start ossart")
    #niter = 30
    #proctime = time.perf_counter()
    #image = tigre.algorithms.ossart(ims,geo,angles,niter,blocksize=20)
    #print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_tigre_ossart.nrrd", origin, spacing)
    #print("start sirt")
    #niter = 30
    #proctime = time.perf_counter()
    #image = utils.SIRT_astra(out_shape, geo)(ims, niter, free_memory=True)
    #print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_astra_sirt.nrrd", origin, spacing, True, False)
    #print("start cgls")
    #niter = 15
    #proctime = time.perf_counter()
    #image = utils.CGLS_astra(out_shape, geo)(ims, niter, free_memory=True)
    #print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_astra_cgls.nrrd", origin, spacing, True, False)
    #print("start fista")
    #niter = 70
    #proctime = time.perf_counter()
    #fistaout = tigre.algorithms.fista(ims,geo,angles,niter,hyper=2.e4)
    #image = sitk.GetImageFromArray(fistaout)
    #image.SetOrigin(origin[0])
    #image.SetDirection(origin[1])
    #print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #sitk.WriteImage(image, prefix+"reco_tigre_fista.nrrd", origin, spacing)
    #print("start asd pocs")▲
    #niter = 10
    #proctime = time.perf_counter()
    #out_shape[1] += 20
    #image = utils.ASD_POCS_astra(out_shape, geo)(ims, niter, free_memory=True)
    #print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
    #save_image(image, prefix+"reco_tigre_asd_pocs.nrrd", origin, spacing, True, False)
    if True:
        print("start PICCS")
        astra.clear()
        niter = 100
        proctime = time.perf_counter()
        for i,image in enumerate(piccs.piccs(ims,out_shape,geo,angles,niter,initial,real_image)):
            if type(image) is list:
                save_plot(image, prefix, "piccs")
            else:
                print("Runtime: ", time.perf_counter() - proctime, "Error:", np.mean(np.abs(real_image-image)))
                save_image(image, prefix+"reco_piccs_"+str(i)+".nrrd", origin, spacing, False, False)

def circle_mask(size):
    _xx, yy, zz = np.mgrid[:size[0], :size[1], :size[2]]
    tube = (yy - size[1]/2)**2 + (zz - size[2]/2)**2
    #sitk.WriteImage(sitk.GetImageFromArray(tube), "test.nrrd")
    mask = tube > (size[1]/2 - 20)**2
    #sitk.WriteImage(sitk.GetImageFromArray(np.array(mask,dtype=int)), "mask.nrrd")
    return mask

def save_plot(data, prefix, title):
    data = np.array(data)
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("recos", prefix + "plot_" + title + ".png"))
    with open(os.path.join("recos", prefix + "plot_" + title + ".csv"), "w") as f:
        f.writelines([str(t)+";"+str(v)+"\n" for t,v in data])

def save_image(image, filename, origin, spacing, switch_axes=False, hu_transform=True, crop=False):
    image = np.array(image[:], dtype=float)
    if hu_transform:
        μW = 0.019286726
        μA = 0.000021063006
        mask = circle_mask(image.shape)
        image[mask] = 0.0
        image = np.array(image, dtype=np.float32)
        if crop:
            image = image[20:-20,20:-20,20:-20]
        image = 1000.0*((image - μW)/(μW-μA))
    else:
        #image = image-np.min(image)
        #image[image>np.mean(image)+4*np.std(image)] = np.mean(image)+4*np.std(image)
        #image = image / np.max(image)
        #image = image*100 - 50
        #image[image>5] = 5
        image = image*100
        if crop:
            image = image[20:-20,20:-20,20:-20]
    #name = 'vectors_' + prefix.split('_', maxsplit=1)[1][:-1] + '.mat'
    if switch_axes:
        image = sitk.GetImageFromArray(np.swapaxes(image, 1,2)[::-1,::-1])
    else:
        image = sitk.GetImageFromArray(image)
    if origin is not None:
        image.SetOrigin(origin[0])
        image.SetDirection(origin[1])
    if spacing is not None:
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
    return (origin, direction), size, spacing, image

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
    #('fake_imbu_cbct_', r".\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
    ]

    origin, size, spacing, image = read_cbct_info(r".\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")

    for prefix, path in data:
        print(prefix, path)
        proctime = time.perf_counter()
        try:
            #ims, angles = read_dicoms(path, reg_angles=True)
            #geo = create_geo(ims.shape, size-20, spacing)
            #geo = tigre.geometry_default(high_quality=False)
            #size = size // 6
            #size = np.array([100, 90, 60])
            #geo.nDetector = np.array([128, 128])
            #geo.sDetector = geo.nDetector*geo.dDetector
            #spacing = [1.0,1.0,1.0]
            #origin = None
            #spacing = spacing * 10
            #geo.nVoxel = np.roll(size+20, 1)           # number of voxels              (vx)
            #geo.sVoxel = np.roll((size+20)*spacing, 1)    # total size of the image       (mm)
            #geo.dVoxel = np.roll(spacing, 1)
            # define angles
            angles=np.linspace(0,2*np.pi,200,dtype=np.float32)
            angles=np.vstack((angles, 0.2*( np.sin(angles)*np.pi - np.pi*0.5 ), np.zeros_like(angles))).T
            #angles3=np.vstack((np.zeros_like(angles), np.zeros_like(angles), angles)).T
            #angles3=np.vstack((angles, angles, angles)).T
            angles3 = angles
            angles3[:,0] = -angles[:,0]-np.pi*0.5
            #angles3[:,2] = angles[:,2] + np.pi*0.5
            # load head phantom data
            #from tigre.demos.Test_data import data_loader
            #head=data_loader.load_head_phantom(number_of_voxels=geo.nVoxel)*spacing[0]*0.1
            import scipy.io
            head = []
            phant = scipy.io.loadmat('phantom.mat')['phantom256']
            for _ in range(5):
                head += [phant*spacing[0]]*20
                head += [np.zeros_like(phant)]*20
                
            head += [phant*spacing[0]]*20
            head = np.array(head)
            size = np.array(head.shape)
            out_shape = size
            out_shape = np.roll(size, 2)
            save_image(head, prefix+"reco.nrrd", origin, spacing, False, False, False)
            #head = np.swapaxes(np.swapaxes(head, 1,2), 0, 1)
            detector_shape = np.array((len(angles),1920//4,2480//4))
            #save_image(ims, prefix+"reco_sinogram.nrrd", None, None, False, False)
            #head = ((np.array(sitk.GetArrayFromImage(image), dtype=np.float32)/1000)*0.019286726)+0.019286726
            #head[head<0] = 0
            # generate projections
            geo = create_geo(detector_shape, size-20, spacing)
            ims1=tigre.Ax(np.array(head, dtype=np.float32),geo,angles3*np.array([-1,1,1]),'interpolated')
            save_image(ims1, prefix+"reco_tigre_sinogram.nrrd", None, None, False, False)
            dSD = 1198
            dSI = 785
            astra_geo = utils.create_astra_geo(angles, (0.154*1920/detector_shape[1]/spacing[0], 0.154*2480/detector_shape[2]/spacing[0]), detector_shape[1:], dSI/spacing[0], (dSD-dSI)/spacing[0])
            #out_shape = np.roll(size, 1)
            #out_shape = size
            #head = np.swapaxes(np.swapaxes(head, 1,2), 0, 1)
            ims2 = utils.Ax_astra(size, astra_geo)(head)
            save_image(ims2, prefix+"reco_astra_sinogram.nrrd", None, None, False, False)
            #ims3 = utils.Atb_astra(head.shape, astra_geo)(ims)/len(ims)
            #save_image(ims3, prefix+"reco_astra_bp.nrrd", origin, spacing, False)

            #reco(prefix, ims, angles, geo, origin, size, spacing)
            reco_astra(prefix, head, ims2, angles, astra_geo, origin, out_shape, spacing)
        except Exception as e:
            print(str(e))
            raise

        print("Runtime :", time.perf_counter() - proctime)