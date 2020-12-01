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
import numpy as np
import scipy.io
import utils
import SimpleITK as sitk
import os.path

vol_geom = astra.create_vol_geom(256, 256, 1)

angles = np.linspace(0, 2*np.pi, 360,False)
proj_geom_p = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)
proj_geom_c = astra.create_proj_geom('cone', 1.0, 1.0, 128, 192, angles, 1000, 700)
angles3=np.vstack((angles, np.zeros_like(angles), np.zeros_like(angles))).T
#angles3=np.vstack((np.zeros_like(angles), angles, np.zeros_like(angles))).T
#angles3=np.vstack((np.zeros_like(angles), np.zeros_like(angles), angles)).T
proj_geom_v = utils.create_astra_geo(angles3, (1,1), (128, 192), 1000, 700)

# Create a simple hollow cube phantom
cube = np.zeros((128,128,128))
cube[17:113,17:113,17:113] = 1
cube[33:97,33:97,33:97] = 0

cube = np.array([scipy.io.loadmat('phantom.mat')['phantom256']])

# Create projection data from this

volume_id = astra.data3d.create('-vol', vol_geom, cube)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "sino_parallel.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 150)
rec = astra.data3d.get(rec_id)
sitk.WriteImage(sitk.GetImageFromArray(rec), os.path.join("recos", "fdk_parallel.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)

volume_id = astra.data3d.create('-vol', vol_geom, cube)
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
sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "sino_cone.nrrd"))
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 150)
rec = astra.data3d.get(rec_id)
sitk.WriteImage(sitk.GetImageFromArray(rec), os.path.join("recos", "fdk_cone.nrrd"))
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)

volume_id = astra.data3d.create('-vol', vol_geom, cube)
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

sitk.WriteImage(sitk.GetImageFromArray(proj_data), os.path.join("recos", "sino_vec.nrrd"))

# Display a single projection image
import pylab
#pylab.gray()
#pylab.figure(1)
#pylab.imshow(proj_data[:,20,:])
#print(proj_data.shape)

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)
# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 150)

# Get the result
rec = astra.data3d.get(rec_id)
sitk.WriteImage(sitk.GetImageFromArray(rec), os.path.join("recos", "fdk_vec.nrrd"))
#pylab.figure(2)
#pylab.imshow(rec[0,:,:])
#pylab.show()
#print(rec.shape)

#print(cube.shape, angles.shape, proj_geom_p['DetectorRowCount'], proj_geom_p['DetectorColCount'], proj_data.shape, rec.shape)
# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)