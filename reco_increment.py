import astra
import numpy as np
import scipy.io
import scipy.ndimage
import utils
import SimpleITK as sitk
import os.path
import time
import mlem
import test

image_out_mult = 100
astra_iter = 0
astra_algo = "FDK_CUDA"

origin, size, spacing, image = utils.read_cbct_info(r"E:\output\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
target = utils.fromHU(sitk.GetArrayFromImage(image))
origin, size, spacing, image = utils.read_cbct_info(r"E:\output\CKM_LumbalSpine\20201020-151825.858000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV")
prior = utils.fromHU(sitk.GetArrayFromImage(image))

vol_geom = astra.create_vol_geom(prior.shape[1], prior.shape[2], prior.shape[0])

sitk.WriteImage(sitk.GetImageFromArray(target*image_out_mult), os.path.join("recos", "target.nrrd"))
sitk.WriteImage(sitk.GetImageFromArray(prior*image_out_mult), os.path.join("recos", "prior.nrrd"))

detector_shape = np.array([256, 384])
detector_spacing = np.array([1,1])
dist_source_origin = 2000
dist_detector_origin = 200

angles = np.linspace(0, 2*np.pi, 360, False)
angles_zero = np.zeros_like(angles)
angles_one = np.ones_like(angles)
#proj_geom_c = astra.create_proj_geom('cone', detector_spacing[0], detector_spacing[1], detector_shape[0], detector_shape[1], angles-angles_one*0.5*np.pi, dist_source_origin, dist_detector_origin)
angles_astra = np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T

def proj_astra(angles_astra, volume):
    proj_geom_v, _filt = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
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

def reco_astra(proj_data, angles_astra, astra_algo, name, prior):
    if astra_iter == 0: return
    perftime = time.perf_counter()
    #angles_astra=np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
    proj_geom_v, _filt = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
    proj_id = astra.data3d.create('-sino', proj_geom_v, proj_data)

    sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "astra_"+name+"_sino.nrrd"))
    rec_id = astra.data3d.create('-vol', vol_geom, prior)
    cfg = astra.astra_dict(astra_algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, astra_iter)
    rec = astra.data3d.get(rec_id)
    sitk.WriteImage(sitk.GetImageFromArray(rec*image_out_mult), os.path.join("recos", "astra_"+name+"_reco.nrrd"))
    sitk.WriteImage(sitk.GetImageFromArray((prior-rec)*image_out_mult), os.path.join("recos", "astra_"+name+"_change.nrrd"))
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    print(proj_data.shape, detector_shape, angles_astra.shape)
    print("Astra "+name+": ", time.perf_counter()-perftime, np.sum(np.abs(target-rec)))
    return rec


proj_prior = proj_astra(angles_astra, prior)
sitk.WriteImage(sitk.GetImageFromArray(proj_prior), os.path.join("recos", "proj_prior.nrrd"))
proj_target = proj_astra(angles_astra, target)
sitk.WriteImage(sitk.GetImageFromArray(proj_target), os.path.join("recos", "proj_target.nrrd"))

subs = 8
angles_eight = []
#angles_eight.append([180, 0, 0])
for α, β in zip(np.linspace(180, 135, subs, False), np.linspace(0, 30, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([135, 30, 0])
for α, β in zip(np.linspace(135, 90, subs, False), np.linspace(30, 0, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([90, 0, 0])
for α, β in zip(np.linspace(90, 45, subs, False), np.linspace(0, -30, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([45, -30, 0])
for α, β in zip(np.linspace(45, 0, subs, False), np.linspace(-30, 0, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([0, 0, 0])
for α, β in zip(np.linspace(0, 45, subs, False), np.linspace(0, 30, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([45, 30, 0])
for α, β in zip(np.linspace(45, 90, subs, False), np.linspace(30, 0, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([90, 0, 0])
for α, β in zip(np.linspace(90, 135, subs, False), np.linspace(0, -30, subs, False)):
    angles_eight.append([α, β, 0])
#angles_eight.append([135, -30, 0])
for α, β in zip(np.linspace(135, 180, subs), np.linspace(-30, 0, subs)):
    angles_eight.append([α, β, 0])
angles_eight = np.array(angles_eight)*np.pi/180
angles_eight[:,1] = angles_eight[:,1]+np.pi*0.5

subs *= 2
angles_square = []
for α, β in zip(np.linspace(135, 45, subs, False), np.linspace(30, 30, subs, False)):
    angles_square.append([α, β, 0])
for α, β in zip(np.linspace(45, 45, subs, False), np.linspace(30, -30, subs, False)):
    angles_square.append([α, β, 0])
for α, β in zip(np.linspace(45, 135, subs, False), np.linspace(-30, -30, subs, False)):
    angles_square.append([α, β, 0])
for α, β in zip(np.linspace(135, 135, subs, False), np.linspace(-30, 30, subs, False)):
    angles_square.append([α, β, 0])
angles_square = np.array(angles_square)*np.pi/180
angles_square[:,1] = angles_square[:,1]+np.pi*0.5

angles_arc = []
subs *= 4
for α, β in zip(np.linspace(135, 45, subs), np.linspace(0, 0, subs)):
    angles_arc.append([α, β, 0])
angles_arc = np.array(angles_arc)*np.pi/180
angles_arc[:,1] = angles_arc[:,1]+np.pi*0.5

angles_half = []
for α, β in zip(np.linspace(180, 0, subs), np.linspace(0, 0, subs)):
    angles_half.append([α, β, 0])
angles_half = np.array(angles_half)*np.pi/180
angles_half[:,1] = angles_half[:,1]+np.pi*0.5

proj_eight = proj_astra(angles_eight, target)
sitk.WriteImage(sitk.GetImageFromArray(proj_eight), os.path.join("recos", "proj_eight.nrrd"))

proj_square = proj_astra(angles_square, target)
sitk.WriteImage(sitk.GetImageFromArray(proj_square), os.path.join("recos", "proj_square.nrrd"))

proj_arc = proj_astra(angles_arc, target)
sitk.WriteImage(sitk.GetImageFromArray(proj_arc), os.path.join("recos", "proj_arc.nrrd"))

proj_half = proj_astra(angles_half, target)
sitk.WriteImage(sitk.GetImageFromArray(proj_half), os.path.join("recos", "proj_half.nrrd"))

proj_geom_v, _filt = utils.create_astra_geo(angles_eight, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
proj_eight_sorted, sort_order_eight = utils.sort_projs(proj_eight, proj_geom_v)
sitk.WriteImage(sitk.GetImageFromArray(proj_eight_sorted), os.path.join("recos", "proj_eight_sorted.nrrd"))

proj_geom_v, _filt = utils.create_astra_geo(angles_square, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
proj_square_sorted, sort_order_square = utils.sort_projs(proj_square, proj_geom_v)
sitk.WriteImage(sitk.GetImageFromArray(proj_square_sorted), os.path.join("recos", "proj_square_sorted.nrrd"))

proj_geom_v, _filt = utils.create_astra_geo(angles_arc, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
proj_arc_sorted, sort_order_arc = utils.sort_projs(proj_arc, proj_geom_v)
sitk.WriteImage(sitk.GetImageFromArray(proj_arc_sorted), os.path.join("recos", "proj_arc_sorted.nrrd"))

proj_geom_v, _filt = utils.create_astra_geo(angles_half, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
proj_half_sorted, sort_order_half = utils.sort_projs(proj_half, proj_geom_v)
sitk.WriteImage(sitk.GetImageFromArray(proj_half_sorted), os.path.join("recos", "proj_half_sorted.nrrd"))

rec_prior = reco_astra(proj_prior, angles_astra, astra_algo, "rec_prior", 0)
rec_taget = reco_astra(proj_target, angles_astra, astra_algo, "rec_target", 0)
rec_eight = reco_astra(proj_eight, angles_eight, astra_algo, "rec_eight", 0)
rec_square = reco_astra(proj_square, angles_square, astra_algo, "rec_square", 0)
rec_arc = reco_astra(proj_arc, angles_arc, astra_algo, "rec_arc", 0)
rec_half = reco_astra(proj_half, angles_half, astra_algo, "rec_half", 0)

proj_prior_eight = np.hstack((proj_prior, proj_eight, proj_eight))
angles_prior_eight = np.vstack((angles_astra, angles_eight, angles_eight))
rec_prior_and_eight = reco_astra(proj_prior_eight, angles_prior_eight, astra_algo, "rec_prior+eight", prior)
rec_prior_and_eight = reco_astra(proj_prior_eight, angles_prior_eight, "SIRT3D_CUDA", "rec_prior+eight_sirt", prior)

rec_prior_eight = reco_astra(proj_eight, angles_eight, astra_algo, "rec_prior_eight", prior)
rec_prior_eight = reco_astra(proj_eight, angles_eight, "SIRT3D_CUDA", "rec_prior_eight_sirt", prior)


proj_prior_square = np.hstack((proj_prior, proj_square, proj_square))
angles_prior_square = np.vstack((angles_astra, angles_square, angles_square))
rec_prior_and_square = reco_astra(proj_prior_square, angles_prior_square, astra_algo, "rec_prior+square", prior)
rec_prior_and_square = reco_astra(proj_prior_square, angles_prior_square, "SIRT3D_CUDA", "rec_prior+square_sirt", prior)

rec_prior_square = reco_astra(proj_square, angles_square, astra_algo, "rec_prior_square", prior)
rec_prior_square = reco_astra(proj_square, angles_square, "SIRT3D_CUDA", "rec_prior_square_sirt", prior)


proj_prior_arc = np.hstack((proj_prior, proj_arc, proj_arc))
angles_prior_arc = np.vstack((angles_astra, angles_arc, angles_arc))
rec_prior_and_arc = reco_astra(proj_prior_arc, angles_prior_arc, astra_algo, "rec_prior+arc", prior)
rec_prior_and_arc = reco_astra(proj_prior_arc, angles_prior_arc, "SIRT3D_CUDA", "rec_prior+arc_sirt", prior)

rec_prior_arc = reco_astra(proj_arc, angles_arc, astra_algo, "rec_prior_arc", prior)
rec_prior_arc = reco_astra(proj_arc, angles_arc, "SIRT3D_CUDA", "rec_prior_arc_sirt", prior)


proj_prior_half = np.hstack((proj_prior, proj_half, proj_half))
angles_prior_half = np.vstack((angles_astra, angles_half, angles_half))
rec_prior_and_half = reco_astra(proj_prior_half, angles_prior_half, astra_algo, "rec_prior+half", prior)
rec_prior_and_half = reco_astra(proj_prior_half, angles_prior_half, "SIRT3D_CUDA", "rec_prior+half_sirt", prior)

rec_prior_half = reco_astra(proj_half, angles_half, astra_algo, "rec_prior_half", prior)
rec_prior_half = reco_astra(proj_half, angles_half, "SIRT3D_CUDA", "rec_prior_half_sirt", prior)

def run_piple(piple_gen, name, target):
    perftime = time.perf_counter()
    for i, rec in enumerate(piple_gen):
        if type(rec) is list:
            #save_plot(rec, name, name)
            pass
        elif type(rec) is tuple:
            #save_plot(rec[0], name +"_error_", name)
            #save_plot(rec[1], name+"_obj_func_", name)
            #WriteAstraImage(rec[2], os.path.join("recos", name+"_reco.nrrd"))
            #WriteAstraImage(cube_astra-rec[2], os.path.join("recos", name+"_error.nrrd"))
            sitk.WriteImage(sitk.GetImageFromArray(rec[2]*image_out_mult), os.path.join("recos", "astra_piple_"+name+"_reco.nrrd"))
            return rec[2]
        else:
            #sitk.WriteImage(sitk.GetImageFromArray(rec*image_out_mult), os.path.join("recos", name+"_" + str(i) + "_reco.nrrd"))
            #sitk.WriteImage(sitk.GetImageFromArray((cube-rec)*image_out_mult), os.path.join("recos", name+"_"+str(i)+"_error.nrrd"))
            print(i, name+": ", time.perf_counter()-perftime, np.sum(np.abs(target-rec)), np.log(np.sum(np.abs(target-rec))))


iters=30
initial=np.zeros_like(prior)
real_image=target
b=10**4
βp=10**3
βr=10**3
p=1

proj_geom_v, _filt = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
#rec_piple = run_piple(mlem.PIPLE(proj_target, prior.shape, proj_geom_v, angles, iters, initial=initial, real_image=prior, b=b, βp=βp, βr=βr, p=p), "target", target)
rec_piple = run_piple(test.reco(proj_target, proj_geom_v, real_image, prior, iters, b=b, g=1, β=β, p=p), "target", target)

proj_geom_v, _filt = utils.create_astra_geo(angles_square, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
#rec_piple = run_piple(mlem.PIPLE(proj_square, prior.shape, proj_geom_v, angles_square, iters, initial=initial, real_image=prior, b=b, βp=βp, βr=βr, p=p), "square", target)
rec_piple = run_piple(test.reco(proj_square, proj_geom_v, real_image, prior, iters, b=b, g=1, β=β, p=p), "square", target)

proj_geom_v, _filt = utils.create_astra_geo(angles_eight, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
#rec_piple = run_piple(mlem.PIPLE(proj_eight, prior.shape, proj_geom_v, angles_eight, iters, initial=initial, real_image=prior, b=b, βp=βp, βr=βr, p=p), "eight", target)
rec_piple = run_piple(test.reco(proj_eight, proj_geom_v, real_image, prior, iters, b=b, g=1, β=β, p=p), "eight", target)

proj_geom_v, _filt = utils.create_astra_geo(angles_arc, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
#rec_piple = run_piple(mlem.PIPLE(proj_arc, prior.shape, proj_geom_v, angles_arc, iters, initial=initial, real_image=prior, b=b, βp=βp, βr=βr, p=p), "arc", target)
rec_piple = run_piple(test.reco(proj_arc, proj_geom_v, real_image, prior, iters, b=b, g=1, β=β, p=p), "arc", target)

proj_geom_v, _filt = utils.create_astra_geo(angles_half, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, 1)
#rec_piple = run_piple(mlem.PIPLE(proj_half, prior.shape, proj_geom_v, angles_half, iters, initial=initial, real_image=prior, b=b, βp=βp, βr=βr, p=p), "half", target)
rec_piple = run_piple(test.reco(proj_half, proj_geom_v, real_image, prior, iters, b=b, g=1, β=β, p=p), "half", target)
