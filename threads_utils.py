import multiprocessing.shared_memory as sm
from numpy import ndarray, array, zeros, cross, dot, round, arccos, arctan2, cos, sin, pi, squeeze
from numpy.linalg import norm
import astra

def from_shm(meta):
    sm_pos = sm.SharedMemory('pos')
    pos = ndarray(meta['pos'][0], meta['pos'][1], buffer=sm_pos.buf)
    pos.flags.writeable = False
    
    points = [None]*(len(meta.keys())-1)
    descs = [None]*(len(meta.keys())-1)
    shms = []
    shms.append(sm_pos)
    for k in meta.keys():
        if k=='pos': continue
        shm = sm.SharedMemory(k)
        shms.append(shm)
        a = ndarray(meta[k][0], meta[k][1], buffer=shm.buf)
        a.flags.writeable = False
        i = int(k[1:])
        if k[0]=='p':
            points[i] = a
        else:
            descs[i] = a
    
    return (pos, points, descs), shms

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
            params = array([params])
        coord_systems = zeros((len(params), 3, 4), dtype=float)
        for i, (t,u,v) in enumerate(params):
            det = cross(u, v)
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

def coord_systems2vecs(coord_systems, detector_spacing, dist_source_origin, dist_origin_detector, image_spacing):
    vectors = zeros((len(coord_systems), 12))
    prims = []
    secs = []
    
    for i in range(len(coord_systems)):
        x_axis = array(coord_systems[i, :,0])
        y_axis = array(coord_systems[i, :,1])
        z_axis = array(coord_systems[i, :,2])
        iso = coord_systems[i, :,3]#-coord_systems[0,:,3]
      
        x_axis /= norm(x_axis)
        y_axis /= norm(y_axis)

        z_axis /= -norm(z_axis)
        
        x = rotMat(90,z_axis).dot(y_axis)

        if dot(x,x_axis)>0:
            if round(arccos(z_axis.dot([0,1,0]))*180/pi) <= 9:
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
        
        prims.append(arctan2(z_axis[1], z_axis[2]))
        secs.append(arctan2(z_axis[0], z_axis[2]))
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

def rotMat(θ, u_not_normed):
    u = u_not_normed / norm(u_not_normed)
    cT = cos(θ/180*pi)
    sT = sin(θ/180*pi)
    return squeeze(array([
                [cT+u[0]*u[0]*(1-cT), u[0]*u[1]*(1-cT)-u[2]*sT, u[0]*u[2]*(1-cT)+u[1]*sT],
                [u[1]*u[0]*(1-cT)+u[2]*sT, cT+u[1]*u[1]*(1-cT), u[1]*u[2]*(1-cT)-u[0]*sT],
                [u[2]*u[0]*(1-cT)-u[1]*sT, u[2]*u[1]*(1-cT)+u[0]*sT, cT+u[2]*u[2]*(1-cT)]
            ]))
