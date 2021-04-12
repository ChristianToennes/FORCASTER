import astra
import numpy as np
import scipy.io
import scipy.ndimage
import utils
import SimpleITK as sitk
import os.path
import time
import test
import forcast_test
import forcast

tigre_iter = 0
astra_iter = 0
stat_iter = 40
image_out_mult = 100

astra_algo = 'SIRT3D_CUDA'
#tigre_algo = tigre.algorithms.sirt

phant = np.array([scipy.io.loadmat('phantom.mat')['phantom256']], dtype=np.float32)[0]

cube = []
for _ in range(13):
    cube += [np.zeros_like(phant) for _ in range(10)]
    cube += [phant for _ in range(10)]
cube += [np.zeros_like(phant) for _ in range(10)]
cube = np.array(cube)

cube = sitk.GetArrayFromImage(sitk.ReadImage("3D_Shepp_Logan.nrrd"))
if os.path.exists(r"D:\lumbal_spine_13.10.2020\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"):
    origin, size, spacing, image = utils.read_cbct_info(r"D:\lumbal_spine_13.10.2020\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
else:
    origin, size, spacing, image = utils.read_cbct_info(r"E:\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
#_origin, _size, _spacing, prior_image = utils.read_cbct_info(r"D:\lumbal_spine_13.10.2020\output\CKM_LumbalSpine\20201020-151825.858000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
cube = utils.fromHU(sitk.GetArrayFromImage(image))
#prior_cube = utils.fromHU(sitk.GetArrayFromImage(prior_image))
#cube = scipy.ndimage.zoom(np.swapaxes(sitk.GetArrayFromImage(sitk.ReadImage("StanfordBunny.nrrd")), 0, 1), 0.5, order=2)
#print(cube.shape)
cube_tigre = np.array(cube, dtype=np.float32)
sitk.WriteImage(sitk.GetImageFromArray(cube*image_out_mult), os.path.join("recos", "target.nrrd"))
image_shape = np.array(cube.shape)
image_spacing = np.array([1,1,1])
#astra_zoom = np.array([256,256,256])/image_shape
astra_zoom = 1
cube_astra = scipy.ndimage.zoom(cube, astra_zoom, order=2)/np.mean(astra_zoom)
#astra_spacing = np.array(cube_astra.shape) / image_shape
astra_spacing = image_shape / np.array(cube_astra.shape)
#cube_astra = cube
detector_shape = np.array([768, 1024])
detector_spacing = np.array([0.5,0.5])
dist_source_origin = 2000
dist_detector_origin = 400
rec_mult = image_spacing[0]*image_spacing[1]*image_spacing[2]
sino_mult = 1
cube_mult = 1

im_size = max(dist_source_origin * detector_spacing[0]*detector_shape[0]/(dist_source_origin+dist_detector_origin),
              dist_source_origin * detector_spacing[1]*detector_shape[1]/(dist_source_origin+dist_detector_origin))

#image_zoom = im_size / image_shape[0]
#image_zoom = image_shape[0]/im_size
image_zoom = 1/astra_spacing[0]

def WriteAstraImage(im, path): 
    sim = sitk.GetImageFromArray(im*image_out_mult*np.mean(astra_zoom))
    sim.SetSpacing(astra_spacing[[1,2,0]])
    sitk.WriteImage(sim, path)

def WriteTigreImage(im, path): 
    sim = sitk.GetImageFromArray(im*image_out_mult)
    sitk.WriteImage(sim, path)

WriteAstraImage(cube_astra, os.path.join("recos", "astra_target.nrrd"))
#WriteTigreImage(cube_tigre, os.path.join("recos", "tigre_target.nrrd"))
print(cube_astra.shape, cube_tigre.shape)

vol_geom = astra.create_vol_geom(cube_astra.shape[1], cube_astra.shape[2], cube_astra.shape[0])

angles = np.linspace(0, 2*np.pi, 10, False)
angles_zero = np.zeros_like(angles)
angles_one = np.ones_like(angles)

proj_geom_p = astra.create_proj_geom('parallel3d', detector_spacing[0]/image_zoom, detector_spacing[1]/image_zoom, detector_shape[0], detector_shape[1], angles-angles_one*0.5*np.pi)
proj_geom_c = astra.create_proj_geom('cone', detector_spacing[0]/image_zoom, detector_spacing[1]/image_zoom, detector_shape[0], detector_shape[1], angles-angles_one*0.5*np.pi, dist_source_origin/image_zoom, dist_detector_origin/image_zoom)
#angles3=np.vstack((np.zeros_like(angles), angles, np.zeros_like(angles))).T
#angles3=np.vstack((np.zeros_like(angles), np.zeros_like(angles), angles)).T
angles_astra=np.vstack((angles, angles_zero, angles_zero)).T
#proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing/image_zoom, detector_shape, dist_source_origin/image_zoom, dist_detector_origin/image_zoom, image_zoom)

if False:
    perftime = time.perf_counter()
    volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
    proj_id = astra.data3d.create('-sino', proj_geom_p, 0)
    algString = 'FP3D_CUDA'
    cfg = astra.astra_dict(algString)
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = volume_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(volume_id)
    proj_data = astra.data3d.get(proj_id)
    sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_parallel.nrrd"))
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict(astra_algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, astra_iter)
    rec = astra.data3d.get(rec_id)
    WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_parallel.nrrd"))
    WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_parallel.nrrd"))
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    print("Astra Parallel: ", time.perf_counter()-perftime, np.sum(np.abs(cube_astra-rec)))

    angles_tigre_add = np.vstack((angles_one*np.pi*1, -1*angles_one*np.pi*0.5, 1*angles_one*np.pi)).T
    angles_tigre_mult = np.vstack((angles_one, angles_one, -1*angles_one)).T

    perftime = time.perf_counter()
    angles_astra=np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
    angles_tigre = angles_astra*angles_tigre_mult+angles_tigre_add
    proj_data = tigre.Ax(cube_tigre, geo_p, angles_tigre, 'interpolated')
    sitk.WriteImage(sitk.GetImageFromArray(np.moveaxis(proj_data, 0,1)), os.path.join("recos", "tigre_sino_parallel.nrrd"))
    rec = tigre_algo(proj_data, geo_p, angles_tigre, niter=tigre_iter)
    WriteTigreImage(rec, os.path.join("recos", "tigre_sirt_parallel.nrrd"))
    WriteTigreImage(cube_tigre-rec, os.path.join("recos", "tigre_error_parallel.nrrd"))
    print("Tigre Parallel: ", time.perf_counter()-perftime, np.sum(np.abs(cube_tigre-rec)), np.mean(np.abs(cube_tigre-rec)))

    perftime = time.perf_counter()
    volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
    proj_id = astra.data3d.create('-sino', proj_geom_c, 0)
    algString = 'FP3D_CUDA'
    cfg = astra.astra_dict(algString)
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = volume_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(volume_id)
    proj_data = astra.data3d.get(proj_id)
    sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_cone.nrrd"))
    rec_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict(astra_algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, astra_iter)
    rec = astra.data3d.get(rec_id)
    WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_cone.nrrd"))
    WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_cone.nrrd"))
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    print("Astra Cone: ", time.perf_counter()-perftime, np.sum(np.abs(cube_astra-rec)))

def proj_astra(angles_astra, volume):
    proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
    volume_id = astra.data3d.create('-vol', vol_geom, volume)
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
    return proj_data

def reco_astra2(proj_data, angles_astra, name, prior=0):
    if astra_iter == 0: return np.zeros_like(cube_astra)
    perftime = time.perf_counter()
    #angles_astra=np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
    proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
    proj_id = astra.data3d.create('-sino', proj_geom_v, proj_data)

    sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_"+name+"_sino.nrrd"))
    rec_id = astra.data3d.create('-vol', vol_geom, prior)
    cfg = astra.astra_dict(astra_algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, astra_iter)
    rec = astra.data3d.get(rec_id)
    WriteAstraImage(utils.toHU(rec), os.path.join("recos", "astra_"+name+"_hu.nrrd"))
    WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_"+name+"_reco.nrrd"))
    WriteAstraImage(np.abs(cube_astra-rec*rec_mult), os.path.join("recos", "astra_"+name+"_error.nrrd"))
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    print(proj_data.shape, detector_shape, angles_astra.shape)
    print("Astra "+name+": ", time.perf_counter()-perftime, np.sum(np.abs(cube_astra-rec)))
    return rec

def reco_astra(angles_astra, name):
    proj_data = proj_astra(angles_astra, cube_astra*cube_mult)
    return reco_astra2(proj_data, angles_astra, name)

angles_astra=np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
name = "circ"
reco_astra(angles_astra, name)

angles_astra=np.vstack((angles, angles_zero, angles_zero)).T
name = "rot1"
reco_astra(angles_astra, name)

angles_astra=np.vstack((angles_zero, angles, angles_zero)).T
name = "rot2"
reco_astra(angles_astra, name)

angles_astra=np.vstack((angles_zero, angles_zero, angles)).T
name = "rot3"
reco_astra(angles_astra, name)

angles_astra=np.vstack((angles, angles, angles)).T
name = "rotA"
reco_astra(angles_astra, name)

angles_astra=np.vstack((angles, np.sin(3*angles)*(30/180*np.pi)+angles_one*0.5*np.pi, angles_one*np.pi)).T
name = "3sin"
reco_astra(angles_astra, name)

angles_astra=np.vstack((angles, np.sin(2*angles)*(30/180*np.pi)+angles_one*0.5*np.pi, angles_one*np.pi)).T
name = "2sin"
reco_astra(angles_astra, name)


import mlem
import matplotlib.pyplot as plt

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

def reco(algo, name, angles_astra, proj_data, stat_iter):
    if stat_iter == 0: return
    perftime = time.perf_counter()
    proj_geom_v, _filt = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, astra_zoom)
    #sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", name+"_sino.nrrd"))
    for i, rec in enumerate(algo(proj_data, proj_geom_v)):
        if type(rec) is list:
            save_plot(rec, name, name)
        elif type(rec) is tuple:
            save_plot(rec[0], name +"_error_", name)
            save_plot(rec[1], name+"_obj_func_", name)
            WriteAstraImage(rec[2], os.path.join("recos", name+"_reco.nrrd"))
            WriteAstraImage(cube_astra-rec[2], os.path.join("recos", name+"_error.nrrd"))
        else:
            #sitk.WriteImage(sitk.GetImageFromArray(rec*image_out_mult), os.path.join("recos", name+"_" + str(i) + "_reco.nrrd"))
            #sitk.WriteImage(sitk.GetImageFromArray((cube-rec)*image_out_mult), os.path.join("recos", name+"_"+str(i)+"_error.nrrd"))
            print(i, name+": ", time.perf_counter()-perftime, np.sum(np.abs(cube_astra-rec)), np.log(np.sum(np.abs(cube_astra-rec))))
    proj_data_out = proj_astra(angles_astra, rec[2])
    sitk.WriteImage(sitk.GetImageFromArray(proj_data_out), os.path.join("recos", name+"_sino.nrrd"))
    print(name, time.perf_counter()-perftime, np.sum(np.abs(cube_astra-rec[2])), np.log(np.sum(np.abs(cube_astra-rec[2]))))
    astra.clear()
    return rec[2]

angles_astra_clean = np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
image_shape = cube_astra.shape
initial = np.zeros(image_shape)
astra_iter = 100

proj_data_clean = proj_astra(angles_astra_clean, cube_astra)
lam = 0.01*np.ones_like(proj_data_clean)*np.max(proj_data_clean, axis=(0,2))[np.newaxis,:,np.newaxis]
noise = np.array(np.random.poisson(lam=lam, size=proj_data_clean.shape), dtype=float)
proj_data = proj_data_clean + noise
g = np.random.uniform(low=1, high=10, size=len(angles_astra_clean))

sitk.WriteImage(sitk.GetImageFromArray(proj_data_clean), os.path.join("recos", "input_proj_data_clean.nrrd"))
sitk.WriteImage(sitk.GetImageFromArray(forcast.Projection_Preprocessing(proj_data_clean)), os.path.join("recos", "input_proj_data.nrrd"))
#angles_astra += np.random.uniform(low=0, high=0.05, size=angles_astra.shape)
real_geo = utils.create_astra_geo(angles_astra_clean, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, astra_zoom)
angles_astra = np.array(angles_astra_clean)
#angles_astra[:,0:3] += np.random.normal(loc=0, scale=1.0*np.pi/180.0, size=angles_astra_clean[:,0:3].shape)
angles_astra[:,0:3] += 0.5*np.pi/180

geo = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, astra_zoom)

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)

def vec2angle(vecs):
    sec = np.arctan2(np.sqrt(vecs[:,6]**2+vecs[:,7]**2), vecs[:,8])
    prim = np.arctan2(vecs[:,7], vecs[:,6])+np.pi
    thrd = np.arctan2(vecs[:,10], vecs[:,9])
    return np.array([prim, sec, thrd]).T

def cmp_vecs(name, real, new):
    ra = vec2angle(real)
    na = vec2angle(new)
    d = ra-na
    while (d>np.pi).any():
        d[d>np.pi] -= 2*np.pi
    while (d<-np.pi).any():
        d[d<-np.pi] += 2*np.pi

    d *= 180/np.pi
    print(name, np.sum(d**2), np.sum(np.abs(d)), np.mean(d), np.std(d))
    #print(d)

def cmp_corrs(name, real, new):
    d = (real-new)[:,3:6]
    while (d>180).any():
        d[d>180] -= 360
    while (d<-180).any():
        d[d<-180] += 360
    print(name, real-new)
    #print(real)
    #print(new)
    print(np.sum(d[:,0]**2), np.sum(np.abs(d[:,0])), np.mean(d[:,0]), np.std(d[:,0]))
    print(np.sum(d[:,1]**2), np.sum(np.abs(d[:,1])), np.mean(d[:,1]), np.std(d[:,1]))
    print(np.sum(d[:,2]**2), np.sum(np.abs(d[:,2])), np.mean(d[:,2]), np.std(d[:,2]))
    print(np.sum(d**2), np.sum(np.abs(d)), np.mean(d), np.std(d))

#print("err", vec2angle(real_geo['Vectors'])-vec2angle(geo['Vectors']))
#print("err", np.sum( (vec2angle(real_geo['Vectors'])-vec2angle(geo['Vectors']) )**2 ))
cmp_vecs("err", real_geo['Vectors'], geo['Vectors'])

Ax = utils.Ax_param_asta(cube_astra.shape, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_spacing, cube_astra)

params_clean = np.zeros((len(angles_astra_clean), 6), dtype=float)
params_clean[:,3:6] = angles_astra_clean*180/np.pi
params_clean[:,4] *= -1
params_clean[:,5] *= -1

params = np.zeros((len(angles_astra), 6), dtype=float)
params[:,3:6] = angles_astra*180/np.pi
params[:,4] *= -1
params[:,5] *= -1

sitk.WriteImage(sitk.GetImageFromArray(Ax(params_clean)), os.path.join("recos", "input_params_clean.nrrd"))
sitk.WriteImage(sitk.GetImageFromArray(Ax(params)), os.path.join("recos", "input_params.nrrd"))

cmp_corrs("err", params_clean, params)

vecs, corrs = forcast_test.reg_and_reco(cube_astra, np.swapaxes(proj_data,0,1), params, Ax, "cube", 0)
cmp_vecs("rough0", real_geo['Vectors'], vecs)
#cmp_vecs("rough-smooth", real_geo['Vectors'], vecs_smooth)
cmp_corrs("rough0", params_clean, corrs)

vecs, corrs = forcast_test.reg_and_reco(cube_astra, np.swapaxes(proj_data,0,1), params, Ax, "cube", 3)
cmp_vecs("rough1", real_geo['Vectors'], vecs)
#cmp_vecs("rough-smooth", real_geo['Vectors'], vecs_smooth)
cmp_corrs("rough1", params_clean, corrs)

vecs, corrs = forcast_test.reg_and_reco(cube_astra, np.swapaxes(proj_data,0,1), params, Ax, "cube", 4)
cmp_vecs("rough2", real_geo['Vectors'], vecs)
#cmp_vecs("rough-smooth", real_geo['Vectors'], vecs_smooth)
cmp_corrs("rough2", params_clean, corrs)

vecs, corrs = forcast_test.reg_and_reco(cube_astra, np.swapaxes(proj_data,0,1), params, Ax, "cube", 5)
cmp_vecs("rough3", real_geo['Vectors'], vecs)
#cmp_vecs("rough-smooth", real_geo['Vectors'], vecs_smooth)
cmp_corrs("rough3", params_clean, corrs)
exit(0)
vecs, corrs = forcast_test.reg_and_reco(cube_astra, np.swapaxes(proj_data,0,1), params, Ax, "cube", 1)
cmp_vecs("less", real_geo['Vectors'], vecs)
#cmp_vecs("less-smooth", real_geo['Vectors'], vecs_smooth)
cmp_corrs("less", params_clean, corrs)

vecs, corrs = forcast_test.reg_and_reco(cube_astra, np.swapaxes(proj_data,0,1), params, Ax, "cube", 2)
cmp_vecs("bfgs", real_geo['Vectors'], vecs)
#cmp_vecs("bfgs-smooth", real_geo['Vectors'], vecs_smooth)
cmp_corrs("bfgs", params_clean, corrs)
exit(0)

#initial = reco_astra(angles_astra, "initial")
from PotentialFilter import potential_filter as c_ψ
from PotentialFilter import potential_dx_filter as c_δψ
from PotentialFilter import potential_dxdx_filter as c_δδψ
from PotentialFilter import potential_dx_t_filter as c_δψ_t
from PotentialFilter import square_filter as c_sq
from PotentialFilter import square_dx_filter as c_δsq
from PotentialFilter import square_dxdx_filter as c_δδsq
from PotentialFilter import mod_p_norm_filter as c_p_norm
from PotentialFilter import mod_p_norm_dx_filter as c_δp_norm
from PotentialFilter import mod_p_norm_dx_t_filter as c_δp_t_norm
from PotentialFilter import mod_p_norm_dxdx_filter as c_δδp_norm
from PotentialFilter import edge_preserving_filter as c_ψ_edge
from PotentialFilter import edge_preserving_dx_filter as c_δψ_edge
from PotentialFilter import edge_preserving_dx_t_filter as c_δψ_t_edge
δψ = c_δψ
δδψ = c_δδψ

iters = {}
for p in ["{}_wls_{}_0".format(p,n) for p in ["quad",] for n in [0,3,5,6]]:
    iters["TEST"+"-"+str(p)+"_"+str(3)] = 50
    
stat_iter = 0
astra_iter = 0
for e in [3]:
    b = 10**(e)*3
    β = 0
    projs = g[np.newaxis,:,np.newaxis]*b*np.exp(-proj_data_clean)
    projs[projs>4096]=4096
    projs_clean = g[np.newaxis,:,np.newaxis]*b*np.exp(-proj_data_clean)
    #projs_clean[projs_clean>4096]=4096

    t_projs = -np.log(projs/(np.max(g)*b))
    t_projs_approx = -np.log(projs/(np.max(projs)*1.2))
    t_projs_clean = -np.log(projs_clean/(g[np.newaxis,:,np.newaxis]*b))

    sitk.WriteImage(sitk.GetImageFromArray(projs), os.path.join("recos", "input_projs.nrrd"))
    sitk.WriteImage(sitk.GetImageFromArray(projs_clean), os.path.join("recos", "input_projs_clean.nrrd"))
    sitk.WriteImage(sitk.GetImageFromArray(t_projs), os.path.join("recos", "input_sino.nrrd"))
    sitk.WriteImage(sitk.GetImageFromArray(t_projs_approx), os.path.join("recos", "input_sino_approx.nrrd"))
    sitk.WriteImage(sitk.GetImageFromArray(t_projs_clean), os.path.join("recos", "input_sino_clean.nrrd"))

    astra_algo = "FDK_CUDA"
    rec_clean = reco_astra2(t_projs_clean, angles_astra, astra_algo+"_clean", prior_cube)
    rec_noise = reco_astra2(t_projs, angles_astra, astra_algo+"_noise", prior_cube)
    rec_approx = reco_astra2(t_projs_approx, angles_astra, astra_algo+"_approx", prior_cube)
    WriteAstraImage(100*np.abs((rec_clean)-(rec_noise)), os.path.join("recos", "astra_"+astra_algo+"_diff.nrrd"))
    WriteAstraImage(100*np.abs((rec_clean)-(rec_approx)), os.path.join("recos", "astra_"+astra_algo+"_diff_approx.nrrd"))
    astra_algo = "SIRT3D_CUDA"
    rec_clean = reco_astra2(t_projs_clean, angles_astra, astra_algo+"_clean", prior_cube)
    rec_noise = reco_astra2(t_projs, angles_astra, astra_algo+"_noise", prior_cube)
    rec_approx = reco_astra2(t_projs_approx, angles_astra, astra_algo+"_approx", prior_cube)
    WriteAstraImage(100*np.abs((rec_clean)-(rec_noise)), os.path.join("recos", "astra_"+astra_algo+"_diff.nrrd"))
    WriteAstraImage(100*np.abs((rec_clean)-(rec_approx)), os.path.join("recos", "astra_"+astra_algo+"_diff_approx.nrrd"))
    #astra_iter = 100
    astra_algo = "CGLS3D_CUDA"
    rec_clean = reco_astra2(t_projs_clean, angles_astra, astra_algo+"_clean", prior_cube)
    rec_noise = reco_astra2(t_projs, angles_astra, astra_algo+"_noise", prior_cube)
    rec_approx = reco_astra2(t_projs_approx, angles_astra, astra_algo+"_approx", prior_cube)
    WriteAstraImage(100*np.abs((rec_clean)-(rec_noise)), os.path.join("recos", "astra_"+astra_algo+"_diff.nrrd"))
    WriteAstraImage(100*np.abs((rec_clean)-(rec_approx)), os.path.join("recos", "astra_"+astra_algo+"_diff_approx.nrrd"))
    
    for p in ["{}_wls_{}_0".format(p,n) for p in ["huber", "quad"] for n in [0,1,2,3,4,5,6]]:
        for e2 in range(1, 20):
            β = 10**(e2/2)
            #name = "TEST"+str(e)+"-"+str(p)+"_"+str(e2)
            name = "TEST"+"-"+str(p)+"_"+str(e2)
            if name in iters:
                rec_clean = reco(lambda proj, geo: test.reco(proj, geo, cube_astra, prior_cube, iters[name], (g*b)[np.newaxis,:,np.newaxis]*np.ones_like(proj), g*b, β, p, α = 0.3), name+"_clean", angles_astra, projs_clean, iters[name])
                rec_noise = reco(lambda proj, geo: test.reco(proj, geo, cube_astra, prior_cube, iters[name], np.max(g)*b, g*b, β, p, α = 0.3), name+"_noise", angles_astra, projs, iters[name])
                rec_approx = reco(lambda proj, geo: test.reco(proj, geo, cube_astra, prior_cube, iters[name], (1.5*np.max(projs)), g*b, β, p, α = 0.3), name+"_approx", angles_astra, projs, iters[name])
                WriteAstraImage(100*np.abs((rec_clean)-(rec_noise)), os.path.join("recos", "astra_"+name+"_diff.nrrd"))
                WriteAstraImage(100*np.abs((rec_clean)-(rec_approx)), os.path.join("recos", "astra_"+name+"_diff_approx.nrrd"))
                #pass

    stat_iter=30
    for e in [5]:
        for e2 in [4]:
            b = 10**e
            β = 10**e2
            for p in [1.4, 1, 2]:
                name = "PIPLE-"+str(p)+"-"+str(e)+"-"+str(e2)
                #rec_piple = reco(lambda proj, geo: mlem.PIPLE(proj, cube_astra.shape, geo, angles_astra, stat_iter, initial=np.zeros_like(cube_astra), real_image=cube_astra, b=b, βp=β, βr=β, p=p, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True), name, angles_astra, proj_data_clean, stat_iter)
            p = "quad_trl_0_0"
            name = "TEST-"+str(p)+"-"+str(e)+"-"+str(e2)
            #rec_clean = reco(lambda proj, geo: test.reco(proj, geo, cube_astra, stat_iter, (g*b)[np.newaxis,:,np.newaxis]*np.ones_like(proj), g*b, β, p, α = 0.3), name+"_clean", angles_astra, proj_data_clean, stat_iter)