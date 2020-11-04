#!/usr/bin/env python
import itk
from itk import RTK as rtk
from reco import load_image
import numpy as np

(img, angles, angles2, dist_source, dist_detector, kv, ms, As, mAs) = load_image(True)

out_file = "reco.nrrd"
out_file_geo = "cbct_geo"
# Defines the image type
GPUImageType = rtk.CudaImage[itk.F,3]
CPUImageType = rtk.Image[itk.F,3]

# Defines the RTK geometry object
geometry = rtk.T
numberOfProjections = len(img)

for phi, theta, sid, did in zip(angles, angles2, dist_source, dist_detector):
  geometry.AddProjectionInRadians(sid,sid+did,phi,outOfPlaneAngle=theta)

# Writing the geometry to disk
xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
xmlWriter.SetFilename (out_file_geo)
xmlWriter.SetObject ( geometry )
xmlWriter.WriteFile()

# Create a stack of empty projection images
ConstantImageSourceType = rtk.ConstantImageSource[GPUImageType]
constantImageSource = ConstantImageSourceType.New()
origin = [ -127, -127, 0. ]
sizeOutput = [ 128, 128,  numberOfProjections ]
spacing = [ 2.0, 2.0, 2.0 ]
constantImageSource.SetOrigin( origin )
constantImageSource.SetSpacing( spacing )
constantImageSource.SetSize( sizeOutput )
constantImageSource.SetConstant(0.)

REIType = rtk.RayEllipsoidIntersectionImageFilter[CPUImageType, CPUImageType]
rei = REIType.New()
semiprincipalaxis = [ 50, 50, 50]
center = [ 0, 0, 10]
# Set GrayScale value, axes, center...
rei.SetDensity(2)
rei.SetAngle(0)
rei.SetCenter(center)
rei.SetAxis(semiprincipalaxis)
rei.SetGeometry( geometry )
rei.SetInput(constantImageSource.GetOutput())

# Create reconstructed image
constantImageSource2 = ConstantImageSourceType.New()
sizeOutput = [ 128, 128, 128 ]
origin = [ -63.5, -63.5, -63.5 ]
spacing = [ 1.0, 1.0, 1.0 ]
constantImageSource2.SetOrigin( origin )
constantImageSource2.SetSpacing( spacing )
constantImageSource2.SetSize( sizeOutput )
constantImageSource2.SetConstant(0.)

# Graft the projections to an itk::CudaImage
projections = GPUImageType.New()
rei.Update()
projections.SetPixelContainer(rei.GetOutput().GetPixelContainer())
projections.CopyInformation(rei.GetOutput())
projections.SetBufferedRegion(rei.GetOutput().GetBufferedRegion())
projections.SetRequestedRegion(rei.GetOutput().GetRequestedRegion())

# FDK reconstruction
print("Reconstructing...")
FDKGPUType = rtk.CudaFDKConeBeamReconstructionFilter
feldkamp = FDKGPUType.New()
feldkamp.SetInput(0, constantImageSource2.GetOutput())
feldkamp.SetInput(1, projections)
feldkamp.SetGeometry(geometry)
feldkamp.GetRampFilter().SetTruncationCorrection(0.0)
feldkamp.GetRampFilter().SetHannCutFrequency(0.0)

# Field-of-view masking
FOVFilterType = rtk.FieldOfViewImageFilter[CPUImageType, CPUImageType]
fieldofview = FOVFilterType.New()
fieldofview.SetInput(0, feldkamp.GetOutput())
fieldofview.SetProjectionsStack(rei.GetOutput())
fieldofview.SetGeometry(geometry)

# Writer
print("Writing output image...")
WriterType = rtk.ImageFileWriter[CPUImageType]
writer = WriterType.New()
writer.SetFileName(out_file)
writer.SetInput(fieldofview.GetOutput())
writer.Update()

print("Done!")