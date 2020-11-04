import numpy as np
import pydicom
import os
import os.path
import skimage.transform

def load_image(InitialCBCT=False, dir_path = "TrajOpti"):
    img = []
    angles = []
    angles2 = []
    dist_source = []
    dist_detector = []
    kv = []
    ms = []
    As = []
    mAs = []
    if InitialCBCT:
        dicom_file = pydicom.read_file("InitialCBCT.dcm")
        dist_source_detector = int(dicom_file[0x0018,0x1110].value)
        dist_source_all = int.from_bytes(dicom_file[0x0021,0x1017].value, "little")
        kv_all = dicom_file[0x0018,0x0060].value
        ms_all = dicom_file[0x0018,0x1154].value / 1000.0
        #ms_all = dicom_file[0x0018,0x1150].value / 1000.0
        As_all = dicom_file[0x0018,0x1151].value
        for img_part in dicom_file.pixel_array:
            img.append(img_part)
            dist_source.append(dist_source_all)
            dist_detector.append(dist_source_detector - dist_source_all)
            kv.append(kv_all)
            ms.append(ms_all)
            As.append(As_all)
            mAs.append(ms_all*As_all)
        angles = np.array(dicom_file[0x0018,0x1520].value)*np.pi / 180
        angles2 = np.array(dicom_file[0x0018,0x1521].value)*np.pi / 180
    else:
        for f in os.listdir(dir_path):
            img_part = pydicom.read_file(os.path.join(dir_path, f))
            img.append(skimage.transform.rescale( img_part.pixel_array, 0.5 ) )
            #img.append(img_part.pixel_array)
            angles.append(img_part[0x0018,0x1510].value*np.pi/180)
            angles2.append(img_part[0x0018,0x1511].value*np.pi/180)
            dist_source_detector = img_part[0x0018,0x1110].value
            dist_source.append(img_part[0x0018,0x1111].value)
            dist_detector.append(dist_source_detector - dist_source[-1])
            kv.append(img_part[0x0018,0x0060].value)
            ms.append(img_part[0x0018,0x1150].value / 1000.0)
            As.append(img_part[0x0018,0x1151].value)
            mAs.append(img_part[0x0018,0x1151].value * img_part[0x0018,0x1150].value / 1000.0)


    img = np.array(img)
    angles = np.array(angles)
    angles2 = np.array(angles2)
    dist_source = np.array(dist_source)
    dist_detector = np.array(dist_detector)
    kv = np.array(kv)
    ms = np.array(ms)
    As = np.array(As)
    mAs = np.array(mAs)

    f1 = kv==np.median(kv) 
    #f2 = np.array([a not in angles[f1][:i] for i, a in enumerate(angles[f1]) ]) | np.array([a not in angles2[f1][:i] for i, a in enumerate(angles2[f1]) ])
    f2 = f1[f1]
    f = f1[:]
    f[f1] = f2
    img = img[f]
    angles = angles[f]
    angles2 = angles2[f]
    dist_source = dist_source[f]
    dist_detector = dist_detector[f]
    kv = kv[f]
    ms = ms[f]
    As = As[f]
    mAs = mAs[f]
    print("filtered", len(img), len(f), np.count_nonzero(~f1), np.count_nonzero(~f2), np.count_nonzero(~f))
    return img, angles, angles2, dist_source, dist_detector, kv, ms, As, mAs


def normalize(images, mAs_array, kV_array):

    kVs = np.array([40, 50, 60, 70, 80, 90, 100, 109, 120, 125])
    a = np.array([20.4125, 61.6163, 138.4021, 250.8008, 398.963, 586.5949, 794.5124, 1006.1, 1252.2, 1404.2202])
    b = np.array([677.964, 686.4824, 684.1844, 691.9573, 701.1038, 711.416, 729.8813, 750.0054, 791.9865, 796.101])

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
    norm_images = np.zeros((images.shape[0], images.shape[1]-20, images.shape[2]-20), dtype=float)
    for i in range(len(fs)):
        norm_images[i] = -np.log(images[i,10:-10,10:-10] / fs[i])
    print(np.mean(f), np.median(f), np.max(f), np.min(f))
    print(np.mean(images), np.median(images), np.max(images), np.min(images))
    print(np.mean(norm_images), np.median(norm_images), np.max(norm_images), np.min(norm_images))
    return norm_images


if __name__ == "__main__":
    import astra
    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    (raw_img, angles, angles2, dist_source, dist_detector, kv, ms, As, mAs) = load_image(True)

    img = normalize(raw_img, mAs, kv)

    proj_data = np.moveaxis(img, -1, 0)
    print("projection data read", proj_data.shape)

    print("kv: ", np.min(kv), np.max(kv), np.median(kv), np.mean(kv))
    print("ms: ", np.min(ms), np.max(ms), np.median(ms), np.mean(ms))
    print("As: ", np.min(As), np.max(As), np.median(As), np.mean(As))
    print("mAs: ", np.min(mAs), np.max(mAs), np.median(mAs), np.mean(mAs))

    #angles = np.linspace(0, np.pi, 180,False)
    #angles = np.array(img[0x0018,0x1520].value)*np.pi / 180
    #print("angles", angles)
    #angles += np.pi/2
    print("angles", angles)
    #angles2 = np.array(img[0x0018,0x1521].value)*np.pi / 180
    print("angles2", angles2)
    #angles2 = np.zeros_like(angles)
    angles2 += np.pi/2

    #dist_source_detector = int(img[0x0018,0x1110].value)
    #dist_source = int.from_bytes(img[0x0021,0x1017].value, "little")
    #dist_detector = (dist_source_detector - dist_source)
    #print("dist source", dist_source, "dist_detector", dist_detector)

    proj_geom = astra.create_proj_geom('cone', 0.154, 0.154, proj_data.shape[0], proj_data.shape[2], angles, np.mean(dist_source), np.mean(dist_detector))
    positions = []
    spacingX = 0.154*3
    spacingY = 0.154*3
    for phi, theta, dist_source, dist_detector in zip(angles, angles2, dist_source, dist_detector):
        # ( rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
        srcX = np.sin(theta)*np.sin(phi) * dist_source
        srcY = -np.sin(theta)*np.cos(phi) * dist_source
        srcZ = np.cos(theta) * dist_source
        dX = -np.sin(theta)*np.sin(phi) * dist_detector
        dY = np.sin(theta)*np.cos(phi) * dist_detector
        dZ = np.cos(theta)*dist_detector
        vX = -np.cos(theta)*np.sin(phi) * spacingX
        vY = np.cos(theta)*np.cos(phi) * spacingX
        vZ = -np.sin(theta)*spacingX
        uX = -np.sin(theta)*np.cos(phi) * spacingY
        uY = -np.sin(theta)*np.sin(phi) * spacingY
        uZ = np.cos(theta)*spacingY
        positions.append(np.array([ srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ ]))
    positions = np.array(positions)


    proj_geom = astra.create_proj_geom('cone_vec', proj_data.shape[0], proj_data.shape[2], positions)
    print("cone beam geometry created")

    proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data)
    print("projector created")
    # Create a simple hollow cube phantom
    #cube = np.zeros((256,256,256))
    #cube[17:113,17:113,17:113] = 1
    #cube[33:97,33:97,33:97] = 0

    # Create projection data from this
    #proj_id, proj_data2 = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
    #astra.data3d.store(proj_id, proj_data)

    # Display a single projection image

    plt.figure(1)
    plt.gray()
    plt.title("proj data")
    plt.imshow(proj_data[:,20,:])

    vol_geom = astra.create_vol_geom(256, 256, 256)
    print("volume created")
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)
    print("reconstruction volume created")
    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('FDK_CUDA')
    #cfg = astra.astra_dict('CGLS3D_CUDA')
    #cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {"ShortScan": True, "VoxelSuperSampling": 3}

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    print("created algorithm", cfg)
    # Run 150 iterations of the algorithm
    # Note that this requires about 750MB of GPU memory, and has a runtime
    # in the order of 10 seconds.
    iterations = 100
    astra.algorithm.run(alg_id, iterations)
    print("algorithm run finished after iterations:", iterations)

    # Get the result
    rec = astra.data3d.get(rec_id)
    print("copied results", rec.shape)
    sitk.WriteImage(sitk.GetImageFromArray(rec), "reco.nrrd")
    print("saved results")
    plt.figure(2)
    plt.title("reconstructed slice")
    plt.gray()
    plt.imshow(rec[rec.shape[0]//2,:,:])

    plt.figure(3)
    plt.title("real slice")
    plt.gray()
    target_img = pydicom.read_file("100252.000000_197.dcm")
    plt.imshow(target_img.pixel_array)

    #print(positions[0][6:9],positions[0][9:])
    #fig = plt.figure(4)
    #ax = fig.gca(projection='3d')
    #for p in positions:
    #    u = p[3:6]
    #    v = p[6:9]
    #    print( np.round( np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v), -1, 1 ) ) * 180 / np.pi ))
    #    v = p[9:]
    #    print( np.round(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v), -1, 1 ) ) * 180 / np.pi ))
    #    u = p[6:9]
    #    print( np.round(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v), -1, 1 ) ) * 180 / np.pi ))
        #print(np.linalg.norm(p[6:9]), np.linalg.norm(p[9:12]))
        #ax.quiver(p[3], p[4], p[5], -p[3]/np.sum(p[3:6]) , -p[4]/np.sum(p[3:6]), -p[5]/np.sum(p[3:6]))
        #ax.plot( [p[3], p[3]+p[6]], [p[4], p[4]+p[7]], [p[5], p[5]+p[8]], color="r")
        #ax.plot( [p[3], p[3]+p[9]], [p[4], p[4]+p[10]], [p[5], p[5]+p[11]], color="b")
    #ax.quiver(positions[0,3], positions[0,4], positions[0,5], positions[0,9], positions[0,10], positions[0,11])

    plt.figure(4)
    plt.title("kV")
    plt.plot(kv, list(range(len(kv))))

    #plt.figure(5)
    #plt.title("ms")
    #plt.plot(ms, list(range(len(ms))))

    plt.figure(5)
    plt.title("mAs")
    plt.plot(mAs, list(range(len(mAs))))

    plt.show()


    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
