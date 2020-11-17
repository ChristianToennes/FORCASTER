import numpy as np
import astra

def Ax_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[0], out_shape[2])
    vol_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_Ax(x, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        astra.data3d.store(vol_id, np.swapaxes(x, 0,2))
        astra.algorithm.run(alg_id, iterations)
        result = np.swapaxes(astra.data3d.get(proj_id), 0,1)
        if free_memory:
            free()
        return result
    return run_Ax

def Atb_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[0], out_shape[2])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_Atb(x, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        astra.data3d.store(proj_id, np.swapaxes(x, 0,1))
        astra.algorithm.run(alg_id, iterations)
        result = np.swapaxes(astra.data3d.get(rec_id), 0,2)
        if free_memory:
            free()
        return result
    return run_Atb

def At2b_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[0], out_shape[2])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_At2b(x, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        astra.data3d.store(proj_id, np.swapaxes(x, 0,1))
        astra.algorithm.run(alg_id, iterations)
        result = np.swapaxes(astra.data3d.get(rec_id), 0,2)
        if free_memory:
            free()
        return result
    return run_At2b

def FDK_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[0], out_shape[2])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {"VoxelSuperSampling": 3}
    alg_id = astra.algorithm.create(cfg)
    iterations = 1
    freed = False
    def free():
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_FDK(x, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        astra.data3d.store(proj_id, np.swapaxes(x, 0,1))
        astra.algorithm.run(alg_id, iterations)
        result = np.array(np.swapaxes(astra.data3d.get(rec_id), 0,2), dtype=float)
        if free_memory:
            free()
        return result
    return run_FDK

def CGLS_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[0], out_shape[2])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('CGLS3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {"VoxelSuperSampling": 1}
    alg_id = astra.algorithm.create(cfg)
    freed = False
    def free():
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_CGLS(x, iterations, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        astra.data3d.store(proj_id, np.swapaxes(x, 0,1))
        astra.algorithm.run(alg_id, iterations)
        result = np.swapaxes(astra.data3d.get(rec_id), 0,2)
        if free_memory:
            free()
        return result
    return run_CGLS


def SIRT_astra(out_shape, proj_geom):
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    vol_geom = astra.create_vol_geom(out_shape[1], out_shape[0], out_shape[2])
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {"VoxelSuperSampling": 1, "MinConstraint": 0}
    alg_id = astra.algorithm.create(cfg)
    freed = False
    def free():
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        freed = True
    def run_SIRT(x, iterations, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        astra.data3d.store(proj_id, np.swapaxes(x, 0,1))
        astra.algorithm.run(alg_id, iterations)
        result = np.swapaxes(astra.data3d.get(rec_id), 0,2)
        if free_memory:
            free()
        return result
    return run_SIRT

def ASD_POCS_astra(out_shape, proj_geom): # sidky 2008

    β = 1
    β_red = 0.995
    ng = 20
    α = 0.2
    r_max = 0.95
    α_red = 0.95
    f = np.ones(out_shape, dtype=np.float32)
    ε = 0.1
    first_iter = True
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    M = astra.OpTomo(proj_id)

    freed = False
    def free():
        astra.data3d.delete(proj_id)
        freed = True    
    def grad_minTV(f):
        dftv = np.zeros_like(f)
        ε = 0.0001
        df = np.moveaxis(np.array([f[1:,:,:] - f[:-1,:,:], f[:,1:,:] - f[:,:-1,:], f[:,:,1:] - f[:,:,:-1]]), 0, -1)
        df2 = df*df
        dftv = np.sum(df[:-1,:-1,:-1], axis=-1) / (np.sqrt(np.sum(df2[:-1,:-1,:-1], axis=-1))+ε) \
            -df[1:,:-1,:-1,0] / (np.sqrt(np.sum(df2[1:,:-1,:-1], axis=-1))+ε) \
            -df[:-1,1:,:-1,1] / (np.sqrt(np.sum(df2[:-1,1:,:-1], axis=-1))+ε) \
            -df[:-1,:-1,1:,2] / (np.sqrt(np.sum(df2[:-1,:-1,1:], axis=-1))+ε)
        return dftv
    def run_ASD_POCS(x, iterations, free_memory=False):
        if freed:
            print("data structures and algorithm already deleted")
            return
        f_res = f
        g̃_0 = x
        for iter_n in range(iterations):
            f_0 = f
            f = f + β*M*((g̃_0-M*f) / (M.dot(M)))# ART
            f[f<0] = 0
            f_res = f
            g̃ = M*f
            dd = np.linalg.norm(g̃ - g̃_0, 2)
            dp = np.linalg.norm(f - f_0, 2)
            if first_iter:
                first_iter = False
                dtvg = α*dp
            f_0 = f
            for i in range(ng):
                df = grad_minTV(f)
                df̂ = df / np.linalg.norm(df)
                f = f - dtvg * df̂
            dg = np.linalg.norm(f-f_0, 2)
            if dg > r_max*dp and dd > ε:
                dtvg = dtvg * α_red
            β = β*β_red

        if free_memory:
            free()
        return f_res
    return run_ASD_POCS

def create_astra_geo(angles, detector_spacing, detector_size, dist_source_origin, dist_origin_detector):
    vectors = np.zeros((len(angles), 12))

    for i in range(len(angles)):
        
        ## source
        #vectors[i,1] = np.sin(angles[i][0]) * dist_source_origin
        #vectors[i,2] = -np.cos(angles[i][0]) * dist_source_origin
        #vectors[i,3] = 0

        ## center of detector
        #vectors[i,4] = -np.sin(angles[i][0]) * dist_origin_detector
        #vectors[i,5] = np.cos(angles[i][0]) * dist_origin_detector
        #vectors[i,6] = 0

        # vector from detector pixel (0,0) to (0,1)
        #vectors[i,7] = np.cos(angles[i][0]) * detector_spacing[0]
        #vectors[i,8] = np.sin(angles[i][0] ) * detector_spacing[0]
        #vectors[i,9] = 0

        ## vector from detector pixel (0,0) to (1,0)
        #vectors[i,10] = 0
        #vectors[i,11] = 0
        #vectors[i,12] = detector_spacing[1]

        γ, β, α = angles[i]
        γ = -γ - np.pi*0.5
        #β = β + np.pi*0.5
        #β = -β
        α = α + np.pi
        cα, cβ, cγ = np.cos(-α), np.cos(-β), np.cos(-γ)
        sα, sβ, sγ = np.sin(-α), np.sin(-β), np.sin(-γ)

        Rz = lambda x: np.array([ [np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1] ])
        Ry = lambda x: np.array([ [np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)] ])
        Rx = lambda x: np.array([ [1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)] ])

        R = np.array([
            [cα*cβ, cα*sβ*sγ-sα*cγ, cα*sβ*cγ+sα*sγ],
            [sα*cβ, sα*sβ*sγ+cα*cγ, sα*sβ*cγ-cα*sγ],
            [-sβ, cβ*sγ, cβ*cγ]
        ])

        #R = Rx(γ).dot(Ry(β).dot(Rz(α)) ) 

        srcX, srcY, srcZ = R.dot([0,0,-dist_source_origin])
        dX, dY, dZ = R.dot([0,0,dist_origin_detector])
        vX, vY, vZ = R.dot([detector_spacing[0], 0, 0])
        uX, uY, uZ = R.dot([0,detector_spacing[1], 0])

        vectors[i] = srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ

    # Parameters: #rows, #columns, vectors
    proj_geom = astra.create_proj_geom('cone_vec', detector_size[0], detector_size[1], vectors)
    return proj_geom
