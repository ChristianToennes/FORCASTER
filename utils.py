import numpy as np
import astra
import os
import SimpleITK as sitk
import time
import matplotlib.pyplot as plt
from itertools import zip_longest
import cv2
from multiprocessing import shared_memory as sm

default_config = {"use_cpu": True, "AKAZE_params": {"threshold": 0.0005, "nOctaves": 4, "nOctaveLayers": 5, 
    "descriptor_type": cv2.AKAZE_DESCRIPTOR_MLDB, "descriptor_size": 0, "descriptor_channels": 3, "diffusivity": cv2.KAZE_DIFF_PM_G2},
"my": True, "grad_width": (1,25), "noise": None, "both": False, "max_change": 1}

def FP(img, dist_detector_source, ratio):
    proj_geom = astra.create_proj_geom('cone', 0.25, 0.25, 2480, 1920, [0, np.pi*0.5, np.pi, np.pi*1.5], dist_detector_source*ratio, dist_detector_source*(1.0-ratio)) 
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(img.shape[1], img.shape[2], img.shape[0])
    vol_id = astra.data3d.create('-vol', vol_geom, img)
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    result = astra.data3d.get(proj_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)
    return result

def Ax_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    vol_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_Ax(x, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        astra.data3d.store(vol_id, x)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(proj_id)
        if free_memory:
            free()
        return result
    run_Ax.free = free
    return run_Ax

def Ax_geo_astra(out_shape, x):
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    vol_id = astra.data3d.create('-vol', vol_geom, x)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(vol_id)
        freed = True
    def run_Ax(proj_geom, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        proj_id = astra.data3d.create('-proj3d', proj_geom)
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectionDataId'] = proj_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(proj_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)
        if free_memory:
            free()
        return result
    run_Ax.free = free
    return run_Ax

def Ax_vecs_astra(out_shape, detector_shape, x):
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    vol_id = astra.data3d.create('-vol', vol_geom, x)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(vol_id)
        freed = True
    def run_Ax(vecs, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        proj_geom = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        proj_id = astra.data3d.create('-proj3d', proj_geom)
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectionDataId'] = proj_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(proj_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)
        if free_memory:
            free()
        return result
    run_Ax.free = free
    return run_Ax

def Ax_param_asta(out_shape, detector_spacing, detector_size, dist_source_origin, dist_origin_detector, image_spacing, x, super_sampling=1):
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    vol_id = astra.data3d.create('-vol', vol_geom, x)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(vol_id)
        freed = True
    def create_coords(params):
        if len(params.shape) == 1:
            params = np.array([params])
        coord_systems = np.zeros((len(params), 3, 4), dtype=float)
        for i, (t,u,v) in enumerate(params):
            det = np.cross(u, v)
            coord_systems[i,:,0] = v
            coord_systems[i,:,1] = u
            coord_systems[i,:,2] = det
            coord_systems[i,:,3] = t
        return coord_systems
    def create_vecs(params):
        return coord_systems2vecs(create_coords(params), detector_spacing, dist_source_origin, dist_origin_detector, image_spacing)
    def create_geo(params):
        return create_astra_geo_coords(create_coords(params), detector_spacing, detector_size, dist_source_origin, dist_origin_detector, image_spacing)
    def run_Ax(params, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        proj_geom = create_geo(params)
        proj_id = astra.data3d.create('-proj3d', proj_geom)
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectionDataId'] = proj_id
        cfg['VolumeDataId'] = vol_id
        cfg['Option'] = {'DetectorSuperSampling', super_sampling}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(proj_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)
        if free_memory:
            free()
        return result
    run_Ax.free = free
    run_Ax.create_coords = create_coords
    run_Ax.create_vecs = create_vecs
    run_Ax.create_geo = create_geo
    run_Ax.distance_source_origin = dist_source_origin
    return run_Ax

def Ax2_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    vol_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = vol_id
    cfg['Option'] = {"SquaredWeights": True}
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_Ax2(x, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        astra.data3d.store(vol_id, x)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(proj_id)
        if free_memory:
            free()
        return result
    run_Ax2.free = free
    return run_Ax2

def Atb_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_Atb(x, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        astra.data3d.store(proj_id, x)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(rec_id)
        if free_memory:
            free()
        return result
    run_Atb.free = free
    return run_Atb

def At2b_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['Option'] = {"SquaredWeights": True}
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_At2b(x, free_memory=False):
        nonlocal freed
        if freed:
            print("data structures and algorithm already deleted")
            return 0
        astra.data3d.store(proj_id, x)
        astra.algorithm.run(alg_id, iterations)
        result = astra.data3d.get(rec_id)
        if free_memory:
            free()
        return result
    run_At2b.free = free
    return run_At2b

def FDK_astra(out_shape, proj_geom, x):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['Option'] = {#"VoxelSuperSampling": 3, 
                     "ShortScan": False
                    }
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    astra.data3d.store(proj_id, x)
    astra.algorithm.run(alg_id, iterations)
    result = astra.data3d.get(rec_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    astra.algorithm.delete(alg_id)
    return result

def CGLS_astra(out_shape, proj_geom, x, iterations):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('CGLS3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    #cfg['Option'] = {"VoxelSuperSampling": 3}
    alg_id = astra.algorithm.create(cfg)
    astra.data3d.store(proj_id, x)
    astra.algorithm.run(alg_id, iterations)
    result = astra.data3d.get(rec_id)
    result[result==np.nan] = 0
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    astra.algorithm.delete(alg_id)

    return result


def SIRT_astra(out_shape, proj_geom, x, iterations):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    #cfg['Option'] = {"VoxelSuperSampling": 3}
    alg_id = astra.algorithm.create(cfg)
    astra.data3d.store(proj_id, x)
    astra.algorithm.run(alg_id, iterations)
    result = astra.data3d.get(rec_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    astra.algorithm.delete(alg_id)

    return result

def ASD_POCS_astra(out_shape, proj_geom): # sidky 2008
    β = 1
    β_red = 0.995
    ng = 20
    α = 0.2
    r_max = 0.95
    α_red = 0.95
    ε = 0.1
    first_iter = True
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
    proj_id = astra.create_projector('cuda3d',proj_geom, vol_geom)
    M = astra.OpTomo(proj_id)
    f_0 = np.ones(out_shape, dtype=np.float32)
    #f_0 = np.moveaxis(f_0, 2, 0)
    f_shape = f_0.shape
    freed = False
    def free():
        nonlocal freed
        astra.data3d.delete(proj_id)
        freed = True    
    def grad_minTV(f_in):
        dftv = np.zeros_like(f_in)
        ε = 0.0001
        df1 = f_in[1:,1:,1:] - f_in[:-1,1:,1:]
        df2 = f_in[1:,1:,1:] - f_in[1:,:-1,1:]
        df3 = f_in[1:,1:,1:] - f_in[1:,1:,:-1]
        df = np.array([df1, df2, df3])
        df = np.moveaxis(df, 0, -1)
        df2 = df*df

        dftv[1:-1,1:-1,1:-1] = np.sum(df[:-1,:-1,:-1], axis=-1) / (np.sqrt(np.sum(df2[:-1,:-1,:-1], axis=-1))+ε) \
            -df[1:,:-1,:-1,0] / (np.sqrt(np.sum(df2[1:,:-1,:-1], axis=-1))+ε) \
            -df[:-1,1:,:-1,1] / (np.sqrt(np.sum(df2[:-1,1:,:-1], axis=-1))+ε) \
            -df[:-1,:-1,1:,2] / (np.sqrt(np.sum(df2[:-1,:-1,1:], axis=-1))+ε)
        return dftv
    def run_ASD_POCS(x, iterations, free_memory=False):
        nonlocal f_0, freed, first_iter, β
        if freed:
            print("data structures and algorithm already deleted")
            return
        f_res = f_0
        f = f_0
        g̃_0 = x[:]
        for _iter_n in range(iterations):
            f_0 = f
            nom = g̃_0-M.FP(f)
            denom = M.dot(M.T)
            denom = (denom*np.ones(denom.shape[0], dtype=nom.dtype)).reshape(nom.shape) + 0.0001
            f = f + β*M.BP(nom / denom) # ART

            f[f<0] = 0
            f_res = f.reshape(f_shape)
            g̃ = M*f
            dd = np.linalg.norm(g̃ - g̃_0.flatten(), 2)
            dp = np.linalg.norm(f.flatten() - f_0.flatten(), 2)
            if first_iter:
                first_iter = False
                dtvg = α*dp
            f_0 = f
            for _i in range(ng):
                df = grad_minTV(f.reshape(f_shape))
                df̂ = df / np.linalg.norm(df.flatten())
                f = f - dtvg * df̂
            dg = np.linalg.norm((f-f_0).flatten(), 2)
            if dg > r_max*dp and dd > ε:
                dtvg = dtvg * α_red
            β = β*β_red

        if free_memory:
            free()
        f_res = f_res
        return f_res
    run_ASD_POCS.free = free
    return run_ASD_POCS

def tv_norm(x):
    y_diff = x - np.roll(x, -1, axis=0)
    x_diff = x - np.roll(x, -1, axis=1)
    z_diff = x - np.roll(x, -1, axis=2)
    grad_norm2 = x_diff**2 + y_diff**2 + z_diff**2
    norm = np.sum(np.sqrt(grad_norm2))
    return norm


def δtv_norm(x):
    y_diff = x - np.roll(x, -1, axis=0)
    x_diff = x - np.roll(x, -1, axis=1)
    z_diff = x - np.roll(x, -1, axis=2)
    grad_norm2 = x_diff**2 + y_diff**2 + z_diff**2 + 1e-12
    dgrad_norm = 0.5 / np.sqrt(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    dz_diff = 2 * z_diff * dgrad_norm
    grad = dx_diff + dy_diff + dz_diff
    grad[:, 1:, :] -= dx_diff[:, :-1, :]
    grad[1:, :, :] -= dy_diff[:-1, :, :]
    grad[:, :, 1:] -= dz_diff[:, :, :-1]
    return grad

def read_matlab_vecs(prefix):
    name = 'vectors_' + prefix + '.mat'
    if os.path.isfile(name):
        from scipy.io import loadmat
        mat = loadmat(name)
        vecs = mat['newVectors']

        #r = [rotMat(-90, v) for v in np.cross(vecs[:, 6:9], vecs[:, 9:12])]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])

        #r = [rotMat(0, [0,0,1]) for v in vecs[:, 6:9]]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])

        #r = [rotMat(180, v) for v in vecs[:, 9:12]]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])

        #r = [rotMat(0, [1,0,0]) for v in vecs[:, 9:12]]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])


        #a = np.array(vecs[:, 6:9])
        #vecs[:, 6:9] = -vecs[:, 9:12]
        #vecs[:, 9:12] = -a
        r = [rotMat(90, v) for v in np.cross(vecs[:, 6:9], vecs[:, 9:12])]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = -np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])

        #r = [rotMat(180, v) for v in vecs[:, 9:12]]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])
        #vecs[:, 6:9] = vecs[:,6:9] * np.array([-1,-1,-1])[np.newaxis,:]
        #vecs[:, 9:12] = vecs[:,9:12] * np.array([1,1,1])[np.newaxis,:]

        #print(vecs[0])
        #exit()
        return vecs
    else:
        raise FileNotFoundError(name)

def params_to_vecs(params, sid=807, did=391):
    r = rotMat(90, [0,0,1]).dot(rotMat(-90, [1,0,0]))
    vecs = np.zeros((len(params), 12), dtype=float)
    vecs[:, 6:9] = np.array([r.dot(v) for v in params[:,1]])
    vecs[:, 9:12] = np.array([r.dot(v) for v in params[:,2]])
    zdir = np.cross(vecs[:, 6:9], vecs[:, 9:12])
    zdir = zdir/np.linalg.norm(zdir, axis=1)[:,np.newaxis]
    vecs[:, 0:3] = params[:,0] + zdir*sid
    vecs[:, 3:6] = params[:,0] + zdir*did
    return vecs

def vecs_to_params(vecs, sid=807, did=391):
    r = rotMat(90, [1,0,0]).dot(rotMat(-90, [0,0,1]))
    params = np.zeros((len(vecs), 3, 3), dtype=float)
    params[:,1] = np.array([r.dot(v) for v in vecs[:, 6:9]])
    params[:,2] = np.array([r.dot(v) for v in vecs[:, 9:12]])
    zdir = np.cross(vecs[:,9:12], vecs[:,6:9])
    zdir = zdir/np.linalg.norm(zdir, axis=1)[:,np.newaxis]
    i1 = vecs[:, 0:3] + zdir*sid
    i2 = vecs[:, 3:6] + zdir*did
    print(np.linalg.norm(vecs[:5, 0:3]-vecs[:5,3:6], axis=1))
    print(np.linalg.norm(i1[:5]-i2[:5], axis=1))
    print(i1[:5])
    print(i2[:5])
    #print(np.mean(i1, axis=0))
    #print(np.mean(i2, axis=0))
    #params[:,0] = i1
    params[:,0] = i2
    return params

def test():
    vecs = read_matlab_vecs("genA_trans")
    params = vecs_to_params(vecs)
    vecs2 = params_to_vecs(params)
    diff = vecs-vecs2
    
    _, tn, zn = get_noise()

    #print(tn[:5,0][:,np.newaxis]*vecs[:5,6:9]+tn[:5,1][:,np.newaxis]*vecs[:5,9:12])
    #print(params[:5,0])
    #print(diff[:5,0:3])
    #print(diff[:5,3:6])

def get_noise(len_ims=496):
    random = np.random.default_rng(23)
    angles_noise = random.uniform(low=-1, high=1, size=(len_ims,3))
    min_trans, max_trans = -10, 10
    trans_noise = np.array(np.round(random.uniform(low=min_trans, high=max_trans, size=(len_ims,2))), dtype=int)
    zoom_noise = random.uniform(low=0.9, high=1, size=len_ims)

    return angles_noise, trans_noise, zoom_noise


def create_default_astra_geo():
    angles = np.linspace(0, 2*np.pi, 360, False)
    angles_one = np.ones_like(angles)
    angles_astra = np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
    detector_shape = np.array([128, 128])
    detector_spacing = np.array([1,1])
    dist_source_origin = 2000
    dist_detector_origin = 200
    astra_zoom = 1
    return create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, astra_zoom)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s==0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def create_astra_geo(angles, translation, detector_spacing, detector_size, dist_source_origin, dist_origin_detector, image_spacing):
    vectors = np.zeros((len(angles), 12))

    for i in range(len(angles)):
        α, β, γ = angles[i]
        cα, cβ, cγ = np.cos(α), np.cos(β), np.cos(-0)
        sα, sβ, sγ = np.sin(α), np.sin(β), np.sin(-0)
        R = np.array([
            [cα*cβ, cα*sβ*sγ-sα*cγ, cα*sβ*cγ+sα*sγ],
            [sα*cβ, sα*sβ*sγ+cα*cγ, sα*sβ*cγ-cα*sγ],
            [-sβ, cβ*sγ, cβ*cγ]
        ])

        #spher = lambda r: np.array([r*cβ*sα,r*sβ*sα,r*cα])
        #srcX, srcY, srcZ = spher(-dist_source_origin*image_spacing)
        #dX, dY, dZ = spher(dist_origin_detector*image_spacing)
        #R = rotation_matrix_from_vectors(np.array([0,0,1]), np.array([dX,dY,dZ]))
        srcX, srcY, srcZ = R.dot([0,0,-dist_source_origin*image_spacing])
        dX, dY, dZ = R.dot([0,0,dist_origin_detector*image_spacing])

        R = rotMat(γ*180/np.pi, [dX,dY,dZ]).dot(R)

        vX, vY, vZ = R.dot([detector_spacing[0]*image_spacing, 0, 0])
        uX, uY, uZ = R.dot([0,detector_spacing[1]*image_spacing, 0])

        if translation is not None:
            if len(translation.shape) == 2:
                srcX += translation[i][0]
                srcY += translation[i][1]
                srcZ += translation[i][2]

                dX += translation[i][0]
                dY += translation[i][1]
                dZ += translation[i][2]
            elif len(translation.shape) == 1:
                srcX += translation[i]
                srcY += translation[i]
                srcZ += translation[i]

                dX += translation[i]
                dY += translation[i]
                dZ += translation[i]
                
        vectors[i] = srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ
    #print(np.linalg.norm([vX, vY, vZ]), np.linalg.norm([uX, uY, uZ]), np.linalg.norm([dX, dY, dZ]), np.linalg.norm([srcX, srcY, srcZ]), dist_source_origin, dist_origin_detector)

    # Parameters: #rows, #columns, vectors
    proj_geom = astra.create_proj_geom('cone_vec', detector_size[0], detector_size[1], vectors)
    return proj_geom

def rotMat(θ, u_not_normed):
    u = u_not_normed / np.linalg.norm(u_not_normed)
    cT = np.cos(θ/180*np.pi)
    sT = np.sin(θ/180*np.pi)
    return np.squeeze(np.array([
                [cT+u[0]*u[0]*(1-cT), u[0]*u[1]*(1-cT)-u[2]*sT, u[0]*u[2]*(1-cT)+u[1]*sT],
                [u[1]*u[0]*(1-cT)+u[2]*sT, cT+u[1]*u[1]*(1-cT), u[1]*u[2]*(1-cT)-u[0]*sT],
                [u[2]*u[0]*(1-cT)-u[1]*sT, u[2]*u[1]*(1-cT)+u[0]*sT, cT+u[2]*u[2]*(1-cT)]
            ]))

def angles2coord_system(angles):
    coords = np.zeros((len(angles), 3, 4))
    for i,(α,β,γ) in enumerate(angles):
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        
        m = rotMat(-α*180.0/np.pi, x)
        x = m.dot(x)
        y = m.dot(y)
        z = m.dot(z)
        
        m = rotMat(β*180/np.pi, y)
        x = m.dot(x)
        y = m.dot(y)
        z = m.dot(z)

        coords[i,:,0] = x/np.linalg.norm(x)
        coords[i,:,1] = y/np.linalg.norm(y)
        coords[i,:,2] = z/np.linalg.norm(z)
    return coords

def coord_systems2vecs(coord_systems, detector_spacing, dist_source_origin, dist_origin_detector, image_spacing):
    vectors = np.zeros((len(coord_systems), 12))
    prims = []
    secs = []
    
    #shift = np.array([0,0,0.63])

    for i in range(len(coord_systems)):
        x_axis = np.array(coord_systems[i, :,0])
        y_axis = np.array(coord_systems[i, :,1])
        z_axis = np.array(coord_systems[i, :,2])
        iso = coord_systems[i, :,3]#-coord_systems[0,:,3]
      
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)

        z_axis /= -np.linalg.norm(z_axis)
        
        x = rotMat(90,z_axis).dot(y_axis)

        if np.dot(x,x_axis)>0:
            if np.round(np.arccos(z_axis.dot([0,1,0]))*180/np.pi) <= 9:
                z_axis = rotMat(180, y_axis).dot(z_axis)
            else:
                z_axis = rotMat(180, y_axis).dot(z_axis)
            #iso -= shift
        #else:
        #    iso += shift
            
        x_axis *= detector_spacing[0]*image_spacing
        vX, vY, vZ = x_axis
        y_axis *= detector_spacing[1]*image_spacing
        uX, uY, uZ = y_axis
        
        prims.append(np.arctan2(z_axis[1], z_axis[2]))
        secs.append(np.arctan2(z_axis[0], z_axis[2]))
        v_detector = z_axis * dist_origin_detector*image_spacing + iso*image_spacing
        v_source = -z_axis * dist_source_origin*image_spacing + iso*image_spacing

        srcX, srcY, srcZ = v_source
        dX, dY, dZ = v_detector

        vectors[i] = srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ
        
    return vectors

def create_astra_geo_coords(coord_systems, detector_spacing, detector_size, dist_source_origin, dist_origin_detector, image_spacing, flips=False):
    vectors = coord_systems2vecs(coord_systems, detector_spacing, dist_source_origin, dist_origin_detector, image_spacing)
    
    #np.savetxt("coords.csv", vectors, delimiter=",")
    proj_geom = astra.create_proj_geom('cone_vec', detector_size[0], detector_size[1], vectors)
    return proj_geom

test_data = "044802004201113212fa0002f96ffbfeff4c04fe06fa00ae0431039600980000000000008000000000ffff0402040002010080008000800080008000800080008000800c000080008000800080008000800080ffffffffff0200800400ffff5aff1000cd0300000101060900000202020276008aff5aff10000000ff0cd3ffee0000800080000000807f0988009d0400800080008011030080add20000f62a32000000000014baffff01000000c40900000600000059d2ffff070000009d2a000008000000eaffffff09000000000000002a000000f8ffffff2b0000002c0000002c000000fdffffff36000000dfffffff370000000be3ffff38000000e1220000390000001419f3ff3a0000001bfcffff3b000000b87d0c00e80300002d0200003e000000000000003f00000000000000d007000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a02c673f43e3a4bad0f2dbbe420991c4662011bc5cf57fbf568c80bc68250fbc15e7dbbe2241933cbf2367bf007b89440000c07f0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f0000c07f01fe7f3fce02b73a52a9fb3b778bafc45988b7baedff7f3f7a71063af48554c13ea3fbbb1e4209ba0ffe7f3f5de675443ea3fb3b1e42093a0ffe7fbf778bafc401fe7f3fce02b73a52a9fb3bf48554c15988b73aedff7fbf7a7106ba5de67544c3f5a8be8f4294c27b94b242ec1d04c61f851fc148d1ff45000000000000000000"

def unpack_sh_stparm(data, f = "12h5h6ch8hcc5h2h6c5h2h4chc3c4h3cc17hh50i12f12f12f12f6fh7c"):
    import struct

    #"12h5h6ch8hcc5h2h6c5h2h4chc3c4h3cc17hh50i12f12f12f12f6fh7c"

    length = struct.calcsize(f)
    data_list = struct.unpack(f, data[-length:])

    data_dict = {}
    i=0
    data_dict["CRAN_A"] = data_list[i]
    i+=1
    data_dict["RAO_A"] = data_list[i]
    i+=1
    data_dict["X_A"] = data_list[i]
    i+=1
    data_dict["Y_A"] = data_list[i]
    i+=1
    data_dict["Z_A"] = data_list[i]
    i+=1
    data_dict["ANGUL_A"] = data_list[i]
    i+=1
    data_dict["ORBITAL_A"] = data_list[i]
    i+=1
    data_dict["SID_A"] = data_list[i]
    i+=1
    data_dict["SOD_A"] = data_list[i]
    i+=1
    data_dict["TOD"] = data_list[i]
    i+=1
    data_dict["MAGN_A"] = data_list[i]
    i+=1
    data_dict["MFTV_A"] = data_list[i]
    i+=1
    data_dict["ROTAT_A"] = data_list[i]
    i+=1
    data_dict["ROTAT2_A"] = data_list[i]
    i+=1
    data_dict["ROTAT3_A"] = data_list[i]
    i+=1
    data_dict["MSOF_A"] = data_list[i]
    i+=1
    data_dict["COLLTV_A"] = data_list[i]
    i+=1
    data_dict["INV_IMG_DEFAULT_A"] = data_list[i]
    i+=1
    data_dict["MFIELD_POSSIBLE_A"] = data_list[i]
    i+=1
    data_dict["INV_IMG_DISPLAY_A"] = data_list[i]
    i+=1
    data_dict["IMG_ROT_A"] = data_list[i]
    i+=1
    data_dict["PLANE_A_PARK"] = data_list[i]
    i+=1
    data_dict["GRID_STATUS_A"] = data_list[i]
    i+=1
    
    data_dict["CRAN_B"] = data_list[i]
    i+=1
    data_dict["RAO_B"] = data_list[i]
    i+=1
    data_dict["X_B"] = data_list[i]
    i+=1
    data_dict["Y_B"] = data_list[i]
    i+=1
    data_dict["Z_B"] = data_list[i]
    i+=1
    data_dict["ANGUL_B"] = data_list[i]
    i+=1
    data_dict["ORBITAL_B"] = data_list[i]
    i+=1
    data_dict["SID_B"] = data_list[i]
    i+=1
    data_dict["SOD_B"] = data_list[i]
    i+=1
    data_dict["TBL_TOP"] = data_list[i]
    i+=1
    i+=1 # DUMMY1
    data_dict["MAGN_B"] = data_list[i]
    i+=1
    data_dict["MFTV_B"] = data_list[i]
    i+=1
    data_dict["ROTAT_B"] = data_list[i]
    i+=1
    data_dict["ROTAT2_B"] = data_list[i]
    i+=1
    data_dict["ROTAT3_B"] = data_list[i]
    i+=1
    data_dict["MSOF_B"] = data_list[i]
    i+=1
    data_dict["COLLTV_B"] = data_list[i]
    i+=1
    data_dict["INV_IMG_DEFAULT_B"] = data_list[i]
    i+=1
    data_dict["MFIELD_POSSIBLE_B"] = data_list[i]
    i+=1
    data_dict["PLANE_A_PARK"] = data_list[i]
    i+=1
    data_dict["INV_IMG_DISPLAY_B"] = data_list[i]
    i+=1
    data_dict["IMG_ROT_B"] = data_list[i]
    i+=1
    data_dict["DIS_ALL_TBL_VALUES"] = data_list[i]
    i+=1
    data_dict["SYS_TILT"] = data_list[i]
    i+=1
    data_dict["TBL_TILT"] = data_list[i]
    i+=1
    data_dict["TBL_ROT"] = data_list[i]
    i+=1
    data_dict["TBL_X"] = data_list[i]
    i+=1
    data_dict["TBL_Y"] = data_list[i]
    i+=1
    data_dict["TBL_Z"] = data_list[i]
    i+=1
    data_dict["TBL_CRADLE"] = data_list[i]
    i+=1
    data_dict["PAT_POS"] = data_list[i]
    i+=1
    data_dict["STPAR_ACT"] = data_list[i]
    i+=1
    data_dict["SYS_TYPE"] = data_list[i]
    i+=1
    data_dict["TBL_TYPE"] = data_list[i]
    i+=1
    data_dict["POSNO"] = data_list[i]
    i+=1
    data_dict["MOVE_A"] = data_list[i]
    i+=1
    data_dict["MOVE_B"] = data_list[i]
    i+=1
    data_dict["MOVE_TBL"] = data_list[i]
    i+=1
    data_dict["DISMODE"] = data_list[i]
    i+=1
    data_dict["IO_HEIGHT"] = data_list[i]
    i+=1
    data_dict["TBL_VERT"] = data_list[i]
    i+=1
    data_dict["TBL_LONG"] = data_list[i]
    i+=1
    data_dict["TBL_LAT"] = data_list[i]
    i+=1
    data_dict["STAND_SPECIAL"] = data_list[i]
    i+=1
    data_dict["ROOM_SELECTION"] = data_list[i]
    i+=1
    data_dict["GRID_STATUS_B"] = data_list[i]
    i+=1
    data_dict["VERSION_SHSTPAR"] = data_list[i]
    i+=1
    data_dict["CAREPOS_X_A"] = data_list[i]
    i+=1
    data_dict["CAREPOS_Y_A"] = data_list[i]
    i+=1
    data_dict["CAREPOS_X_B"] = data_list[i]
    i+=1
    data_dict["CAREPOS_Y_B"] = data_list[i]
    i+=1
    data_dict["CAMERAROT_A"] = data_list[i]
    i+=1
    data_dict["CAMERAROT_B"] = data_list[i]
    i+=1
    data_dict["IT_LONG_A"] = data_list[i]
    i+=1
    data_dict["IT_LAT_A"] = data_list[i]
    i+=1
    data_dict["IT_HEIGHT_A"] = data_list[i]
    i+=1
    data_dict["IT_LONG_B"] = data_list[i]
    i+=1
    data_dict["IT_LAT_B"] = data_list[i]
    i+=1
    data_dict["IT_HEIGHT_B"] = data_list[i]
    i+=1
    data_dict["SISOD_A"] = data_list[i]
    i+=1
    data_dict["SISOD_B"] = data_list[i]
    i+=1
    data_dict["ISO_WORLD_X_A"] = data_list[i]
    i+=1
    data_dict["ISO_WORLD_Y_A"] = data_list[i]
    i+=1
    data_dict["ISO_WORLD_Z_A"] = data_list[i]
    i+=1
    data_dict["NO_ELEM_STAND_PRIV"] = data_list[i]
    i+=1
    data_dict["STAND_PRIV"] = data_list[i:i+50]
    i+=50
    data_dict["COORD_SYS_C_ARM"] = data_list[i:i+12]
    i+=12
    data_dict["COORD_SYS_C_ARM_B"] = data_list[i:i+12]
    i+=12
    data_dict["COORD_SYS_TABLE"] = data_list[i:i+12]
    i+=12
    data_dict["COORD_SYS_PATIENT"] = data_list[i:i+12]
    i+=12
    data_dict["ROBOT_AXES"] = data_list[i:i+6]
    i+=6
    data_dict["TOKEN"] = data_list[i]
    i+=1
    data_dict["AP_ID_REQUESTER"] = data_list[i]
    i+=1
    data_dict["CONFIRM_POSITION"] = data_list[i]
    i+=1

    #print(data_dict["COORD_SYS_C_ARM"])
    #print(data_dict["COORD_SYS_TABLE"])
    #print(data_dict["COORD_SYS_PATIENT"])
    #print(data_dict["MAGN_A"], data_dict["MFTV_A"], data_dict["MSOF_A"], data_dict["COLLTV_A"])
    #print(data_dict["TOD"])
    #print(data_dict["SID_A"], data_dict["SOD_A"], data_dict["SISOD_A"], data_dict["TOD"])

    return data_dict

def vecs2angles_(v1, v2):
    vectors = np.cross(v1, v2)
    dots1 = np.array([np.dot(v, np.array([1,0,0])) for v in vectors])
    dots2 = np.array([np.dot(v, np.array([0,1,0])) for v in vectors])
    dots3 = np.array([np.dot(v, np.array([0,0,1])) for v in vectors])
    l = np.linalg.norm(vectors, axis=1)
    a = np.arccos(dots1/np.linalg.norm(vectors, axis=1))#*180/np.pi
    b = np.arccos(dots2/np.linalg.norm(vectors, axis=1))#*180/np.pi
    c = np.arccos(dots3/np.linalg.norm(vectors, axis=1))#*180/np.pi
    return np.array([a,b,c]).T

def get_iso(vectors, sid, did, image_spacing):
    ns = np.cross(vectors[:,9:12], vectors[:,6:9])
    ns = ns / np.linalg.norm(ns, axis=1)[:,np.newaxis]
    iso1 = (vectors[:,0:3]+ns*sid) / image_spacing
    #iso2 = (vectors[:,3:6]-ns*did) / image_spacing
    return iso1#, iso2

def vecs2angles(v1, v2):
    vectors = np.cross(v2, v1)
    θ = np.arctan2(np.sqrt(vectors[:,0]**2+vectors[:,1]**2), vectors[:,2])+np.pi
    θ[θ>np.pi] -= 2*np.pi
    ϕ = np.arctan2(vectors[:,1], vectors[:,0])

    n = np.array([v / np.linalg.norm(v) for v in vectors])
    
    a = v1 - np.array([n*(n.dot(v)) for v,n in zip(v1, n)])
    b = []
    for α, β in zip(ϕ,θ):
        cα, cβ, sα, sβ = np.cos(α), np.cos(β), np.sin(α), np.sin(β)
        R = np.array([
            [cα*cβ, sα, cα*sβ],
            [sα*cβ, cα, sα*sβ],
            [-sβ, 0, cβ]
        ])
        dX, dY, dZ = R.dot([0,0,-1])
        R = rotMat(0, [dX,dY,dZ]).dot(R)

        b.append(R.dot([1, 0, 0]))
    b = np.array(b)
    #γ = np.arccos(np.array([np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) ) for a,b in zip(a,b)]) )

    #cross = np.cross(a, b)
    #dots = np.array([n.dot(cross) for n,cross in zip(n,cross)])
    #γ[dots < 0] *= -1

    γ = np.array([np.arctan2(np.cross(a,b).dot(n), a.dot(b) ) for a,b,n in zip(a,b,n)])
    
    γ[γ>np.pi] -= 2*np.pi
    γ[γ<-np.pi] += 2*np.pi

    return np.array([ϕ,θ,γ]).T


def get_data(g):
    detector_shape = np.array([768, 1024])*1
    detector_spacing = np.array([0.5,0.5])/1
    dist_source_origin = 2000
    dist_detector_origin = 400
    astra_zoom = 1
    angles = np.linspace(0, 2*np.pi, 400, False)
    angles_zero = np.zeros_like(angles)
    angles_one = np.ones_like(angles)
    random = np.random.default_rng(23)
    angles_astra_clean = np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
    real_geo = create_astra_geo(angles_astra_clean, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, astra_zoom)
    angles_astra = np.array(angles_astra_clean)
    params_clean = np.zeros((len(angles_astra), 3, 3), dtype=float)
    params_clean[:,1] = real_geo['Vectors'][:, 6:9]
    params_clean[:,2] = real_geo['Vectors'][:, 9:12]
    angles_noise = random.normal(loc=0, scale=1.0*np.pi/180.0, size=angles_astra_clean.shape)
    angles_astra = np.array(angles_astra_clean)
    angles_astra[:,0:3] += angles_noise[:,0:3]
    geo = create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, astra_zoom)
    params = np.zeros((len(angles_astra), 3, 3), dtype=float)
    params[:,1] = geo['Vectors'][:, 6:9]
    params[:,2] = geo['Vectors'][:, 9:12]
    for v in list(locals()):
        g[v] = locals()[v]

def read_cbct_info(path):

    # open 2 fds
    if False:
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

    if False:
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

μW = 0.019286726
μA = 0.000021063006

def toHU(image):
    return 1000.0*((image - μW)/(μW-μA))

def fromHU(image):
    return (μW-μA)*image/1000.0 + μW


def sort_projs(projs, geo):
    #edges = (projs[:-1,:,:]-projs[1:,:,:] + projs[:,:,:-1]-projs[:,:,1:])
    #edges = edges>0
    #diff = np.array((edges.shape[1], edges.shape[1]))
    #for i in range(edges.shape[1]):
    #    diff[i] = np.sum(edges[:,:,:]-edges[:,i,:], axis=(0,2))
    vecs = geo['Vectors'][:,0:3]
    dists = np.zeros((vecs.shape[0], vecs.shape[0]))
    #print(geo['Vectors'].shape, vecs.shape, dists.shape, dists[0].shape)
    for i in range(vecs.shape[0]):
        dists[i] = np.linalg.norm(vecs-vecs[i], axis=1)

    #print(dists[0])
    #print(vecs)
    #print(dists[1])

    nn = [0]
    while(len(nn) < vecs.shape[0]):
        for neighbour in np.argsort(dists[nn[-1]]):
            if neighbour in nn: continue
            nn.append(neighbour)
            #print(neighbour, dists[nn[-1]], vecs[nn[-1]])
            break
    
    #return projs[:,nn,:]

    def calc_len(path, dists):
        l = 0
        prev = None
        for u in path:
            if prev is not None:
                l += dists[prev, u]
            prev = u
        return l
    
    len_nn = calc_len(nn, dists)

    τ0 = 1.0 / (len_nn*vecs.shape[0])
    pher = np.ones(dists.shape, dtype=float)*τ0

    β = 2
    α = 0.1
    m = 50
    q0 = 0.9
    
    def ps(ai):
        r = visited[ai][-1]
        den = np.sum([pher[r,s]*(1/dists[r,s])**β for s in visited[ai] if r!=s and dists[r,s]>0])
        if den == 0:
            den = 1
        p = []
        for s in range(dists.shape[0]):
            if s in visited[ai]: p.append(0) 
            elif r==s: p.append(0)
            elif dists[r,s]==0:
                p = np.zeros(dists.shape[0])
                p[s]=1
                return p
            else: p.append(pher[r,s]*(1/dists[r,s])**β / den)
        return p
    its = 5
    for i in range(its):
        visited = [[0] for _ in range(m)]
        for _ in range(vecs.shape[0]-1):
            for ai in range(m):
                q = np.random.random(1)[0]
                if q <= q0:
                    r = visited[ai][-1]
                    p = []
                    for u in range(vecs.shape[0]):
                        if u in visited[ai]: p.append(0)
                        elif u==r: p.append(0)
                        elif dists[r,u] == 0:
                            p = np.zeros(dists.shape[0])
                            p[u]=1
                            break
                        else: p.append(pher[r,u]*(1/dists[r,u])**β)
                    s = np.argmax(p)
                    if s==0:
                        print(p)
                else:
                    p = np.array(ps(ai))
                    pk = np.cumsum(p)
                    pk = pk / pk[-1]
                    t = np.random.random(1)[0]
                    s = np.argmax(pk>t)
                    if s==0:
                        print(p, pk, t, pk>t)
                visited[ai].append(s)
            for ai in range(m):
                pher[visited[ai][-2], visited[ai][-1]] = (1-α)*pher[visited[ai][-2], visited[ai][-1]] + α * τ0

        lens = [calc_len(visited[ai], dists) for ai in range(m)]
        shortest = np.argmin(lens)
        #print(i, shortest, lens[shortest], calc_len(range(dists.shape[0]),dists))
        prev = None
        for s in visited[shortest]:
            if prev is not None:
                pher[prev, s] = (1-α)*pher[prev, s] + α / lens[shortest]
            prev = s
    
    #print(lens[shortest])
    #print(visited[shortest])
    return projs[:,visited[shortest],:], visited[shortest]


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    def print_val(val):
        if np.abs(val) > 0.1:
            print("{}{: .3f}{}".format(bcolors.RED, val, bcolors.END), end=", ")
        elif np.abs(val) < 0.1:
            print("{}{: .3f}{}".format(bcolors.GREEN, val, bcolors.END), end=", ")
        else:
            print("{}{: .3f}{}".format(bcolors.YELLOW, val, bcolors.END), end=", ")


def applyRot(in_params, α, β, γ):
    params = np.array(in_params)
    #print(params, α, β, γ)
    if α!=0:
        params[1] = rotMat(α, params[2]).dot(params[1])
    if β!=0:
        params[2] = rotMat(β, params[1]).dot(params[2])
    if γ!=0:
        u = np.cross(params[1], params[2])
        R = rotMat(γ, u)
        params[1] = R.dot(params[1])
        params[2] = R.dot(params[2])
    return params

def applyTrans(in_params, x, y, z):
    cur = np.array(in_params)
    if x != 0:
        xdir = cur[1]
        cur[0] += x * xdir
    if y != 0:
        ydir = cur[2]
        cur[0] += y * ydir
    if z != 0:
        zdir = np.cross(cur[2], cur[1])
        #zdir = zdir / np.linalg.norm(zdir)
        cur[0] += z * zdir
    return cur


def filt_conf(config):
    d = {"real_img": config["real_img"], "my": config["my"], "use_cpu": config["use_cpu"], "AKAZE_params": config["AKAZE_params"], "comps": config["comps"], "target_sino": config["target_sino"]}#, "p1": None, "absp1": None, "GIoldold": None, "data_real": None, "points_real": None}
    if "p1" in config:
        d["p1"] = config["p1"]
    if "absp1" in config:
        d["absp1"] = config["absp1"]
    if "GIoldold" in config:
        d["GIoldold"] = config["GIoldold"]
    if "data_real" in config:
        d["data_real"] = [None]*len(config["data_real"])
    if "points_real" in config:
        d["points_real"] = [None]*len(config["points_real"])
    if "est_data" in config:
        d["est_data"] = config["est_data"]
    return d

import pickle
def load_est_data():
    path = 'D:\\lumbal_spine_13.10.2020\\recos\\2010201_est_data_39595.dump'
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def into_shm(est_data):
    pos, points, descs = est_data
    meta = {}
    shms = []
    sm_pos = sm.SharedMemory('pos', size=pos.nbytes, create=True)
    sma_pos = np.ndarray(pos.shape, pos.dtype, buffer=sm_pos.buf)
    meta['pos'] = (sma_pos.shape, sma_pos.dtype)
    shms.append(sm_pos)
    sma_pos[:] = pos

    for i,(p,d) in enumerate(zip(points, descs)):
        sm_p = sm.SharedMemory('p'+str(i), size=p.nbytes, create=True)
        sma_p = np.ndarray(p.shape, p.dtype, buffer = sm_p.buf)
        meta['p'+str(i)] = (sma_p.shape, sma_p.dtype)
        shms.append(sm_p)
        sma_p[:] = p
        sm_d = sm.SharedMemory('d'+str(i), size=d.nbytes, create=True)
        sma_d = np.ndarray(d.shape, d.dtype, buffer=sm_d.buf)
        meta['d'+str(i)] = (sma_d.shape, sma_d.dtype)
        shms.append(sm_d)
        sma_d[:] = d
    
    return meta, shms

def from_shm(meta):
    sm_pos = sm.SharedMemory('pos')
    pos = np.ndarray(meta['pos'][0], meta['pos'][1], buffer=sm_pos.buf)
    pos.flags.writeable = False
    
    points = [None]*(len(meta.keys())-1)
    descs = [None]*(len(meta.keys())-1)
    shms = []
    shms.append(sm_pos)
    for k in meta.keys():
        if k=='pos': continue
        shm = sm.SharedMemory(k)
        shms.append(shm)
        a = np.ndarray(meta[k][0], meta[k][1], buffer=shm.buf)
        a.flags.writeable = False
        i = int(k[1:])
        if k[0]=='p':
            points[i] = a
        else:
            descs[i] = a
    
    return (pos, points, descs), shms

def serialize_est_data(est_data):
    return est_data

def unserialize_est_data(est_data):
    return est_data

def _serialize_est_data(est_data):
    projs, points = est_data
    return projs, [([(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp], desc) for kp, desc in points]

def _unserialize_est_data(est_data):
    projs, points = est_data
    return projs, [(np.array([(cv2.KeyPoint(x=p[0][0],y=p[0][1],size=p[1],angle=p[2],response=p[3],octave=p[4],class_id=p[5])) for p in kp],dtype=object), desc) for kp,desc in points]

def minimize_callback(name, obj, args, newline=False):
    if newline:
        with open(name+".csv", "a") as f:
            f.write("\n")
    def callback(x):
        with open(name+".csv", "a") as f:
            f.write("{};".format(obj(x, *args)))
    return callback


def minimize_log(name, starttime, res):
    with open("opti.csv", "a") as f:
        f.write("{};{};{};{};{};{}\n".format(name, time.perf_counter()-starttime, res["success"], res["nit"], res["nfev"], res["njev"]))
    return res

def load_minimize_log():
    files = sorted([f for f in os.listdir() if "bfgs" in f and "csv" in f])
    data = {}
    for filename in files:
        with open(filename) as f:
            key = filename.split()[0]
            if key not in data:
                data[key] = []
                data[key+"p"] = []
            lines = f.readlines()[1:]
            #print(filename, key)
            for i, content in enumerate(lines):
                while len(data[key]) <= i:
                    data[key].append([])
                    data[key+"p"].append([])
                if "err" in filename:
                    linedata = [np.array([float(d[1:-1].split()[0]),float(d[1:-1].split()[1])]) for d in content.strip()[:-1].split(";") if d != ""]
                else:
                    linedata = [float(d) for d in content.strip()[:-1].split(";") if d != ""]
                data[key][i] += linedata
                data[key+"p"][i].append(len(data[key][i])-1)
                #print(len(linedata), end=", ")
            #print("")
    for k in data.keys():
        if k[-1] == "p": continue
        for i in range(len(data[k])):
            data[k][i] = np.array(data[k][i])
    return data

def load_qut_log():
    files = sorted([f for f in os.listdir("logs")])
    data = {}
    for filename in files:
        with open(os.path.join("logs", filename)) as f:
            key = filename.split()[0]
            if key not in data:
                data[key] = [[]]*496
            index = int(filename.split()[-1].split(".")[0])
            data[key][index] = np.array([np.array([float(d[1:-1].split(",")[0]), float(d[1:-1].split(",")[1])]) for d in f.read().split(";") if d != ""])
    
    for key in data.keys():
        filt = np.ones(len(data[key][0]), dtype=bool)
        if key == "33":
            filt[0:2] = False
            filt[3:5] = False
        else:
            filt[0] = False
            filt[2:4] = False

        filt[-3:-1] = False
        filt[-5] = False
        filt[-7] = False
        filt[-9] = False
        filt[-11] = False
        filt[-13] = False
        filt[-15] = False
            
        data[key] = np.array(data[key])[:,filt]
    odata = {}
    for key in data.keys():
        odata[key] = [
            np.mean(data[key], axis=0)
        ]

    return odata


def get_plot_data(suffix):
    labels = {"-44.err": "BFGS - Our Objective", "-58.err": "BFGS - NGI Objective", 
                "--43.err": "BFGS - Our Objective - Reduced Noise", "-57.err": "BFGS - NGI Objective - Reduced Noise",
                "-24.err": "Mixed BFGS - NGI Objective", "-34.err": "Mixed BFGS - Our Objective",
                "42": "FORCASTER - NGI Objective", "33": "FORCASTER - Our Objective", "--45.err": "BFGS - Our Objective",
                }
    colors = {"-44.err": "#550000", "-58.err": "#000055", 
                "-43.err": "#000000", "-57.err": "#000000",
                "-34.err": "#770000", "-24.err": "#000077",
                "33": "#FF0000",  "42": "#0000FF", "-45.err": "#FF0000"}
    ls = {"-44.err": "-", "-58.err": "-", 
            "-43.err": "-.", "-57.err": "-.",
            "-34.err": "--", "-24.err": "--",
            "33": ":",  "42": ":", "-45.err": "-."}

    labels = {k.replace("err", suffix): labels[k] for k in labels}
    colors = {k.replace("err", suffix): colors[k] for k in colors}
    ls = {k.replace("err", suffix): ls[k] for k in ls}
    data = load_minimize_log()
    for k in data.keys():
        fillvalue = data[k][1][-1] if len(data[k][0]) > len(data[k][1]) else data[k][0][-1]
        data[k] = [np.array(list(map(lambda x: 0.5*(x[0]+x[1]), zip_longest(data[k][0], data[k][1], fillvalue=fillvalue))))]
    odata = load_qut_log()
    data.update(odata)
    return data, labels, colors, ls

def plot_log():

    if False:
        data, labels, colors, ls = get_plot_data("ngi")
        plt.figure()
        #for k in data.keys():
        for k in labels.keys():
            if k not in data:
                continue
            #fillvalue = data[k][1][-1] if len(data[k][0]) > len(data[k][1]) else data[k][0][-1]
            #d = [list(map(lambda x: x[0]+x[1], zip_longest(data[k][0], data[k][1], fillvalue=fillvalue)))]
            #d = [data[k][0] + data[k][1]]
            d = data[k]
            for i,values in enumerate(d):
                if "Mixed" in labels[k]:
                    #plt.plot([0]+list(range(4,len(values)+3)), values, label=labels[k]+" " + str(len(values)) + " " + k, color=colors[k], linestyle=ls[k])
                    plt.plot(list(range(len(values))), values, label=labels[k]+" " + str(len(values)) + " " + k, color=colors[k], linestyle=ls[k])
                    plt.scatter(np.array(data[k+"p"][i])-1, np.array(values)[np.array(data[k+"p"][i])-1], color="k", marker="|", s=500)
                    plt.scatter(np.array([1]+data[k+"p"][i]), np.array(values)[np.array([1]+data[k+"p"][i])], color="gray", marker="|", s=500)
                else:
                    plt.plot(list(range(len(values))), values, label=labels[k]+" " + str(len(values)) + " " + k, color=colors[k], linestyle=ls[k])
                    plt.scatter(data[k+"p"][i], np.array(values)[np.array(data[k+"p"][i])], color="k", marker="|", s=500)
                break
        plt.legend()
        plt.title("NGI")
    
    data, labels, colors, ls = get_plot_data("err")

    if False:
        plt.figure()
        #for k in data.keys():
        for k in labels.keys():
            if k not in data:
                continue
            #fillvalue = data[k][1][-1] if len(data[k][0]) > len(data[k][1]) else data[k][0][-1]
            #d = [np.array(list(map(lambda x: x[0]+x[1], zip_longest(data[k][0], data[k][1], fillvalue=fillvalue))))]
            #d = [data[k][0] + data[k][1]]
            d = data[k]
            for i,values in enumerate(d):
                #if i == 0: continue
                values = values[:,0]
                if "Mixed" in labels[k]:
                    #plt.plot([0]+list(range(4,len(values)+3)), values, label=labels[k]+" " + str(len(values)) + " " + k, color=colors[k], linestyle=ls[k])
                    plt.plot(list(range(len(values))), values, label=labels[k], color=colors[k], linestyle=ls[k])
                else:
                    plt.plot(list(range(len(values))), values, label=labels[k], color=colors[k], linestyle=ls[k])
                #break
        plt.legend()
        plt.tight_layout()
        
        plt.title("SSIM")
        
    plt.figure()
    #for k in data.keys():
    for k in labels.keys():
        if k not in data:
            continue
        #fillvalue = data[k][1][-1] if len(data[k][0]) > len(data[k][1]) else data[k][0][-1]
        #d = [np.array(list(map(lambda x: x[0]+x[1], zip_longest(data[k][0], data[k][1], fillvalue=fillvalue))))]
        #d = [data[k][0] + data[k][1]]
        d = data[k]
        for i,values in enumerate(d):
            #if i == 0: continue
            values = values[:,1]
            if "Mixed" in labels[k]:
                #plt.plot([0]+list(range(4,len(values)+3)), values, label=labels[k]+" " + str(len(values)) + " " + k, color=colors[k], linestyle=ls[k])
                plt.plot(list(range(len(values))), values, label=labels[k], color=colors[k], linestyle=ls[k])
            else:
                plt.plot(list(range(len(values))), values, label=labels[k], color=colors[k], linestyle=ls[k])
            #break
    
    plt.legend(fontsize='large')
    plt.xlabel('Iterations')
    plt.ylabel('NRMSE')
    plt.title("NRMSE / Iterations for the 3rd experiment")
    plt.tight_layout()

    plt.show()


def create_circular_mask(shape, center=None, radius=None, radius_off=0, end_off=30):
    l, h, w = shape
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])-10

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.zeros((l,h,w), dtype=bool)
    mask[:] = (dist_from_center <= (radius-radius_off))[np.newaxis,:,:]
    for i in range(30):
        mask[i] = (dist_from_center <= (i/30)*(radius-radius_off))
        mask[-i-1] = (dist_from_center <= (i/30)*(radius-radius_off))

    return mask

def write_rec(geo, ims, filepath, out_rec_meta, mult=1):
    #geo["Vectors"] = geo["Vectors"]*mult
    mult = int(np.round(ims.shape[1] / geo['DetectorRowCount']))
    geo = astra.create_proj_geom('cone_vec', ims.shape[1], ims.shape[2], geo["Vectors"]*mult)
    out_shape = (out_rec_meta[3][0]*mult, out_rec_meta[3][1]*mult, out_rec_meta[3][2]*mult)
    #print(ims.shape, len(geo['Vectors']))
    rec = FDK_astra(out_shape, geo, np.swapaxes(ims, 0,1))
    #mask = np.zeros(rec.shape, dtype=bool)
    mask = create_circular_mask(rec.shape)
    rec = rec*mask
    del mask
    write_images(toHU(rec), filepath, out_rec_meta, mult)
    return
    rec = SIRT_astra(out_shape, geo, np.swapaxes(ims, 0,1), 500)
    #mask = np.zeros(rec.shape, dtype=bool)
    mask = create_circular_mask(rec.shape)
    rec = rec*mask
    del mask
    write_images(toHU(rec), filepath.rsplit('.', maxsplit=1)[0]+"_sirt."+filepath.rsplit('.', maxsplit=1)[1], out_rec_meta, mult)

    rec = CGLS_astra(out_shape, geo, np.swapaxes(ims, 0,1), 50)
    #mask = np.zeros(rec.shape, dtype=bool)
    mask = create_circular_mask(rec.shape)
    rec = rec*mask
    del mask
    write_images(toHU(rec), filepath.rsplit('.', maxsplit=1)[0]+"_cgls."+filepath.rsplit('.', maxsplit=1)[1], out_rec_meta, mult)

def write_images(rec, filepath, out_rec_meta, mult=1):
    rec = sitk.GetImageFromArray(rec)#*100
    rec.SetOrigin(out_rec_meta[0])
    out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    rec.SetSpacing(out_spacing)
    sitk.WriteImage(rec, filepath, True)
    del rec
   
def write_vectors(name, vecs):
    with open("csv\\"+name+".csv", "w") as f:
        f.writelines([",".join([str(v) for v in vec])+"\n" for vec in vecs])

def read_vectors(path):
    with open(path, "r") as f:
        res = np.array([[float(v) for v in l.split(',')] for l in f.readlines().split()])
    return res
