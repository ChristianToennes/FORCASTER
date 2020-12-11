# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra
import tigre
import numpy as np
import scipy.io
import scipy.ndimage
import utils
import SimpleITK as sitk
import os.path
import time

tigre_iter = 25
astra_iter = 25
image_out_mult = 100

astra_algo = 'SIRT3D_CUDA'
tigre_algo = tigre.algorithms.sirt

#cube = np.array([scipy.io.loadmat('phantom.mat')['phantom256']], dtype=np.float32)
cube = sitk.GetArrayFromImage(sitk.ReadImage("3D_Shepp_Logan.nrrd"))
cube_tigre = np.array(cube, dtype=np.float32)
sitk.WriteImage(sitk.GetImageFromArray(cube*image_out_mult), os.path.join("recos", "target.nrrd"))
image_shape = np.array(cube.shape)
image_spacing = np.array([1,1,1])
astra_zoom = np.array([128,256,256])/image_shape
cube_astra = scipy.ndimage.zoom(cube, astra_zoom, order=1)/np.mean(astra_zoom)
astra_spacing = np.array(cube_astra.shape) / image_shape
astra_spacing = image_shape / np.array(cube_astra.shape)
#cube_astra = cube
detector_shape = np.array([256, 512])
detector_spacing = np.array([2,1])
dist_source_origin = 2000
dist_detector_origin = 200
rec_mult = image_spacing[0]*image_spacing[1]*image_spacing[2]
sino_mult = 1
cube_mult = 1

im_size = max(dist_source_origin * detector_spacing[0]*detector_shape[0]/(dist_source_origin+dist_detector_origin),
              dist_source_origin * detector_spacing[1]*detector_shape[1]/(dist_source_origin+dist_detector_origin))

image_zoom = im_size / image_shape[0]
#image_zoom = image_shape[0]/im_size
image_zoom = 1/astra_spacing[0]

def WriteAstraImage(im, path): 
    sim = sitk.GetImageFromArray(im*image_out_mult*astra_spacing[0])
    sim.SetSpacing(astra_spacing)
    sitk.WriteImage(sim, path)

def WriteTigreImage(im, path): 
    sim = sitk.GetImageFromArray(im*image_out_mult)
    sitk.WriteImage(sim, path)

WriteAstraImage(cube_astra, os.path.join("recos", "astra_target.nrrd"))
WriteTigreImage(cube_tigre, os.path.join("recos", "tigre_target.nrrd"))
print(cube_astra.shape, cube_tigre.shape)

vol_geom = astra.create_vol_geom(cube_astra.shape[1], cube_astra.shape[2], cube_astra.shape[0])

angles = np.linspace(0, 2*np.pi, 360, False)
angles_zero = np.zeros_like(angles)
angles_one = np.ones_like(angles)

proj_geom_p = astra.create_proj_geom('parallel3d', detector_spacing[0]/image_zoom, detector_spacing[1]/image_zoom, detector_shape[0], detector_shape[1], angles-angles_one*0.5*np.pi)
proj_geom_c = astra.create_proj_geom('cone', detector_spacing[0]/image_zoom, detector_spacing[1]/image_zoom, detector_shape[0], detector_shape[1], angles-angles_one*0.5*np.pi, dist_source_origin/image_zoom, dist_detector_origin/image_zoom)
#angles3=np.vstack((np.zeros_like(angles), angles, np.zeros_like(angles))).T
#angles3=np.vstack((np.zeros_like(angles), np.zeros_like(angles), angles)).T
angles_astra=np.vstack((angles, angles_zero, angles_zero)).T
#proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing/image_zoom, detector_shape, dist_source_origin/image_zoom, dist_detector_origin/image_zoom, image_zoom)

angles_tigre = np.vstack((angles, np.ones_like(angles)*np.pi*0.5, np.zeros_like(angles))).T
geo_p = tigre.geometry(mode='parallel',nVoxel = image_shape, default=True)
geo_p.nDetector = detector_shape
geo_p.dDetector = detector_spacing
geo_p.sDetector = geo_p.dDetector * geo_p.nDetector
geo_p.DSD = dist_detector_origin + dist_source_origin
geo_p.DSO = dist_source_origin
geo_p.dVoxel = image_spacing
geo_p.sVoxel = geo_p.nVoxel * geo_p.dVoxel

geo_c = tigre.geometry(mode='cone',nVoxel = image_shape, default=True)
geo_c.nDetector = detector_shape
geo_c.dDetector = detector_spacing
geo_c.sDetector = geo_c.dDetector * geo_c.nDetector
geo_c.DSD = dist_detector_origin + dist_source_origin
geo_c.DSO = dist_source_origin
geo_c.dVoxel = image_spacing
geo_c.sVoxel = geo_c.nVoxel * geo_c.dVoxel

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
print("Astra Parallel: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))

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
print("Astra Cone: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_vec.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict(astra_algo)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, astra_iter)
rec = astra.data3d.get(rec_id)
WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_vec.nrrd"))
WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_vec.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
print("Astra Vec: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles, angles_zero, angles_zero)).T
proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_vec_1.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict(astra_algo)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, astra_iter)
rec = astra.data3d.get(rec_id)
WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_vec_1.nrrd"))
WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_vec_1.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
print("Astra Vec: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles_zero, angles, angles_zero)).T
proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_vec_2.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict(astra_algo)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, astra_iter)
rec = astra.data3d.get(rec_id)
WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_vec_2.nrrd"))
WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_vec_2.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
print("Astra Vec: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles_zero, angles_zero, angles)).T
proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_vec_3.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict(astra_algo)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, astra_iter)
rec = astra.data3d.get(rec_id)
WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_vec_3.nrrd"))
WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_vec_3.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
print("Astra Vec: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles, angles, angles)).T
proj_geom_v = utils.create_astra_geo(angles_astra, detector_spacing, detector_shape, dist_source_origin, dist_detector_origin, image_zoom)
volume_id = astra.data3d.create('-vol', vol_geom, cube_astra*cube_mult)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data*sino_mult), os.path.join("recos", "astra_sino_vec_4.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict(astra_algo)
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, astra_iter)
rec = astra.data3d.get(rec_id)
WriteAstraImage(rec*rec_mult, os.path.join("recos", "astra_sirt_vec_4.nrrd"))
WriteAstraImage(cube_astra-rec*rec_mult, os.path.join("recos", "astra_error_4.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
print("Astra Vec: ", time.perf_counter()-perftime, np.sum(np.mean(cube_astra-rec)))


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
print("Tigre Parallel: ", time.perf_counter()-perftime, np.sum(np.mean(cube_tigre-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles, angles_one*0.5*np.pi, angles_one*np.pi)).T
angles_tigre = angles_astra*angles_tigre_mult+angles_tigre_add
proj_data = tigre.Ax(cube_tigre, geo_c, angles_tigre, 'interpolated')
sitk.WriteImage(sitk.GetImageFromArray(np.moveaxis(proj_data, 0,1)), os.path.join("recos", "tigre_sino_cone.nrrd"))
rec = tigre_algo(proj_data, geo_c, angles_tigre, niter=tigre_iter)
WriteTigreImage(rec, os.path.join("recos", "tigre_sirt_cone.nrrd"))
WriteTigreImage(cube_tigre-rec, os.path.join("recos", "tigre_error_cone.nrrd"))
print("Tigre cone: ", time.perf_counter()-perftime, np.sum(np.mean(cube_tigre-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles, angles_zero, angles_zero)).T
angles_tigre = angles_astra*angles_tigre_mult+angles_tigre_add
proj_data = tigre.Ax(cube_tigre, geo_c, angles_tigre, 'interpolated')
sitk.WriteImage(sitk.GetImageFromArray(np.moveaxis(proj_data, 0,1)), os.path.join("recos", "tigre_sino_cone_1.nrrd"))
rec = tigre_algo(proj_data, geo_c, angles_tigre, niter=tigre_iter)
WriteTigreImage(rec, os.path.join("recos", "tigre_sirt_cone_1.nrrd"))
WriteTigreImage(cube_tigre-rec, os.path.join("recos", "tigre_error_cone_1.nrrd"))
print("Tigre cone: ", time.perf_counter()-perftime, np.sum(np.mean(cube_tigre-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles_zero, angles, angles_zero)).T
angles_tigre = angles_astra*angles_tigre_mult+angles_tigre_add
proj_data = tigre.Ax(cube_tigre, geo_c, angles_tigre, 'interpolated')
sitk.WriteImage(sitk.GetImageFromArray(np.moveaxis(proj_data, 0,1)), os.path.join("recos", "tigre_sino_cone_2.nrrd"))
rec = tigre_algo(proj_data, geo_c, angles_tigre, niter=tigre_iter)
WriteTigreImage(rec, os.path.join("recos", "tigre_sirt_cone_2.nrrd"))
WriteTigreImage(cube_tigre-rec, os.path.join("recos", "tigre_error_cone_2.nrrd"))
print("Tigre cone: ", time.perf_counter()-perftime, np.sum(np.mean(cube_tigre-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles_zero, angles_zero, angles)).T
angles_tigre = angles_astra*angles_tigre_mult+angles_tigre_add
proj_data = tigre.Ax(cube_tigre, geo_c, angles_tigre, 'interpolated')
sitk.WriteImage(sitk.GetImageFromArray(np.moveaxis(proj_data, 0,1)), os.path.join("recos", "tigre_sino_cone_3.nrrd"))
rec = tigre_algo(proj_data, geo_c, angles_tigre, niter=tigre_iter)
WriteTigreImage(rec, os.path.join("recos", "tigre_sirt_cone_3.nrrd"))
WriteTigreImage(cube_tigre-rec, os.path.join("recos", "tigre_error_cone_3.nrrd"))
print("Tigre cone: ", time.perf_counter()-perftime, np.sum(np.mean(cube_tigre-rec)))

perftime = time.perf_counter()
angles_astra=np.vstack((angles, angles, angles)).T
angles_tigre = angles_astra*angles_tigre_mult+angles_tigre_add
proj_data = tigre.Ax(cube_tigre, geo_c, angles_tigre, 'interpolated')
sitk.WriteImage(sitk.GetImageFromArray(np.moveaxis(proj_data, 0,1)), os.path.join("recos", "tigre_sino_cone_4.nrrd"))
rec = tigre_algo(proj_data, geo_c, angles_tigre, niter=tigre_iter)
WriteTigreImage(rec, os.path.join("recos", "tigre_sirt_cone_4.nrrd"))
WriteTigreImage(cube_tigre-rec, os.path.join("recos", "tigre_error_cone_4.nrrd"))
print("Tigre cone: ", time.perf_counter()-perftime, np.sum(np.mean(cube_tigre-rec)))