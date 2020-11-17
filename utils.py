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
    def run_Ax(x):
        astra.data3d.store(vol_id, np.swapaxes(x, 0,2))
        astra.algorithm.run(alg_id, iterations)
        return np.swapaxes(astra.data3d.get(proj_id), 0,1)
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
    def run_Atb(x):
        astra.data3d.store(proj_id, np.swapaxes(x, 0,1))
        astra.algorithm.run(alg_id, iterations)
        return np.swapaxes(astra.data3d.get(rec_id), 0,2)
    return run_Atb

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

        srcX, srcY, srcZ = R.dot([0,0,dist_source_origin])
        dX, dY, dZ = R.dot([0,0,-dist_origin_detector])
        vX, vY, vZ = R.dot([detector_spacing[0], 0, 0])
        uX, uY, uZ = R.dot([0,detector_spacing[1], 0])

        vectors[i] = srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ

    # Parameters: #rows, #columns, vectors
    proj_geom = astra.create_proj_geom('cone_vec', detector_size[0], detector_size[1], vectors)
    return proj_geom
