import tomopy
import numpy as np
import SimpleITK as sitk
import pydicom
import math

img = pydicom.read_file("InitialCBCT.dcm")
#proj_data = np.moveaxis(img.pixel_array, -1, 0)
proj_data = img.pixel_array
print("projection data read")

#angles = np.linspace(0, np.pi, 180,False)
angles = np.array(img[0x0018,0x1520].value)*math.pi / 180
print("angles", angles)
angles -= np.min(angles)
print("angles", angles)
angles2 = img[0x0018,0x1521].value

dist_source_detector = img[0x0018,0x1110].value
dist_source = int.from_bytes(img[0x0021,0x1017].value, "little")
dist_detector = (dist_source_detector - dist_source)
print("dist source", dist_source, "dist_detector", dist_detector)

proj_norm = tomopy.minus_log(proj_data)

rot_center = tomopy.find_center(proj_norm, angles)
print("rot center", rot_center)

# Display a single projection image
import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(proj_data[:,20,:])

rec = tomopy.recon(proj_norm, angles, center=rot_center, )

sitk.WriteImage(sitk.GetImageFromArray(rec), "reco.nrrd")
print("saved results")
pylab.figure(2)
pylab.imshow(rec[:,:,128])

pylab.figure(3)
target_img = pydicom.read_file("100252.000000_128.dcm")
pylab.imshow(target_img.pixel_array)

pylab.show()


# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
