
# =================================================================
# Introduction to image registration and all necesarry components:
# - https://itk.org/Doxygen413/html/RegistrationPage.html
# - https://simpleitk.readthedocs.io/en/master/registrationOverview.html


# Code EXAMPLES for image registration in C++ and Python for linear and nonlinear registrations:
# - https://simpleitk.readthedocs.io/en/master/link_examples.html
# =================================================================

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pydicom

def read_save_image(in_path, out_path):
    reader = sitk.ImageSeriesReader()
    files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(in_path, "", True, True, True)
    reader.SetFileNames(files)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    #reader.SetFileNames(list([fnameFixed + "/" + str(f) + ".dcm" for f in reversed(sorted([int(f.split(".")[0]) for f in os.listdir(fnameFixed)]))]))

    image = reader.Execute()

    sitk.WriteImage(image, out_path)
    return image
   
def createMask(image):
    mask = sitk.BinaryThreshold(image, lowerThreshold = 0.0 , upperThreshold = 1.0)
    return mask

if __name__ == '__main__':
    
    # Set filenames    
    #fnameFixed = "E:/Barbara/Bilddaten/Patientendaten/TACE_Patienten/CT_Pat1.nrrd"
    fnameCT36 = "D:\\moliephantom\\test\\received_files_from_PACS\\M2olie_Phantom\\Spezial^01_MOLIEWorkflow (Erwachsener)\\Abdomen nativ  0.5  Br36  3\\20201002-153639.420000"
    fnameCT54 = "D:\\moliephantom\\test\\received_files_from_PACS\\M2olie_Phantom\\Spezial^01_MOLIEWorkflow (Erwachsener)\\Abdomen nativ  0.5  Br54  3\\20201002-153741.743000"
    
    #fnameMoving = "E:/Barbara/Bilddaten/Patientendaten/TACE_Patienten/CBCT_Pat1.nrrd"
    #fnameMoving = "E:/Barbara/Bilddaten/Patientendaten/TACE_Patienten/MR_Pat1.nrrd"
    fnameT2Map = "D:\\moliephantom\\test\\received_files_from_PACS\\MOLIE_Leberphantom\\T2Map_quant"
    fnameT1 = "D:\\moliephantom\\test\\received_files_from_PACS\\MOLIE_Leberphantom\\CKM^Ruttorf\\t1_mprage_sag_p2_iso_1mm\\20201007-170214.679000"
    fnameT2 = "D:\\moliephantom\\test\\received_files_from_PACS\\MOLIE_Leberphantom\\CKM^Ruttorf\\t2_spc_sag_p2_iso_1mm\\20201007-171509.465000"
    fnameEP = "D:\\moliephantom\\test\\received_files_from_PACS\\MOLIE_Leberphantom\\CKM^Ruttorf\\mr_ep2d_diff_rk_b50_400_800_p3_bi_TRACEW_DFC\\1.3.12.2.1107.5.2.19.45005.2020100717025031711230730.0.0.0.nrrd"
    
    #fnameIn1 = "trajtomo_reco_matlab_sin.nrrd"
    #fnameIn2 = "trajtomo_reco_matlab_sin_reg.nrrd"

    #fnameOut = "trajtomo_reco_matlab_sin_reg_reg.nrrd"
    fnameT1Reg = "D:\\moliephantom\\test\\reg_t1.nrrd"
    fnameT2MapReg = "D:\\moliephantom\\test\\reg_t2map.nrrd"
    fnameT2Reg = "D:\\moliephantom\\test\\reg_t2.nrrd"
    fnameEPReg = "D:\\moliephantom\\test\\reg_ep.nrrd"
    fnameT1RegEl = "D:\\moliephantom\\test\\reg_el_t1.nrrd"
    fnameT2MapRegEl = "D:\\moliephantom\\test\\reg_el_t2map.nrrd"
    fnameT2RegEl = "D:\\moliephantom\\test\\reg_el_t2.nrrd"
    fnameEPRegEl = "D:\\moliephantom\\test\\reg_el_ep.nrrd"
    fnameT1PreReg = "D:\\moliephantom\\test\\pre_t1.nrrd"
    fnameT2MapPreReg = "D:\\moliephantom\\test\\pre_t2map.nrrd"
    fnameT2PreReg = "D:\\moliephantom\\test\\pre_t2.nrrd"
    fnameEPPreReg = "D:\\moliephantom\\test\\pre_ep.nrrd"
    fnameCT36Reg = "D:\\moliephantom\\test\\reg_ct_Br36.nrrd"
    fnameCT54Reg = "D:\\moliephantom\\test\\reg_ct_Br54.nrrd"
    images = [(fnameT2Map, fnameT2MapReg), (fnameT1,fnameT1Reg), (fnameT2, fnameT2Reg), (fnameEP, fnameEPReg)]
    
    # Read Images

    #image_in1 = sitk.ReadImage(fnameIn1)
    #image_in2 = sitk.ReadImage(fnameIn2)
    image_ct36 = read_save_image(fnameCT36, fnameCT36Reg)
    image_ct54 = read_save_image(fnameCT54, fnameCT54Reg)
    image_t2map = read_save_image(fnameT2Map, fnameT2MapPreReg)
    image_t1 = read_save_image(fnameT1, fnameT1PreReg)
    image_t2 = read_save_image(fnameT2, fnameT2PreReg)
    image_ep = read_save_image(fnameEP, fnameEPPreReg)

    reader = sitk.ImageSeriesReader()
    files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(fnameEP, "", True, True, True)
    reader.SetFileNames(files[:len(files)//3])
    reader.SetOutputPixelType(sitk.sitkFloat32)
    #reader.SetFileNames(list([fnameFixed + "/" + str(f) + ".dcm" for f in reversed(sorted([int(f.split(".")[0]) for f in os.listdir(fnameFixed)]))]))
    image_ep = reader.Execute()
    sitk.WriteImage(image_ep, fnameEPPreReg)
    image_ep = sitk.ReadImage(fnameEP, sitk.sitkFloat32)
    print(image_ep.GetSize())
    extract = sitk.ExtractImageFilter()
    extract.SetIndex([0,0,0,0])
    extract.SetSize([image_ep.GetSize()[0],image_ep.GetSize()[1],image_ep.GetSize()[2],0])
    image_ep = extract.Execute(image_ep)
    print(image_ep.GetSize())


    #print("fixed_image", image_ct36.GetSize(), image_ct36.GetPixelIDTypeAsString())
    #print("moving_image", image_t1.GetSize(), image_t1.GetPixelIDTypeAsString())
#    # In case a mask should be used during registration, it can be either read in (as the images above) or created via thresholding:
#    mask = createMask(fixed_image)  # thresholds have to be fixed depending on the target structures


    # Define registration
    registration_method = sitk.ImageRegistrationMethod()
    
    
    image_in1 = image_ct36
    for image_in2, fnameOut in images:
        #image_in2 = image_t2map
        #fnameOut = fnameT2MapReg

        ## It is often appropriate to perform an initialization to generate an initial alignment of the images realized as follows:
        transform = sitk.CenteredTransformInitializer(image_in1,#image_ct36, 
                                                    image_in2,#image_t2map, 
                                                    sitk.Euler3DTransform(), 
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY) # alternative: sitk.CenteredTransformInitializerFilter.MOMENTS
        registration_method.SetInitialTransform(transform)    

        ## Choose similarity metric
        ## (List of similarity measures availible in ITK/SimpleITK: https://itk.org/Doxygen43/html/group__RegistrationMetrics.html)
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)    # multimodal
    #    registration_method.SetMetricAsCorrelation()        # monomodal
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)   # reduces computational demand since the metric isn't calculated in the entire image
        registration_method.SetMetricSamplingPercentage(0.01)
        
    #    # If a mask should betaken into account for the metric calculation during the registration process
    #    registration_method.SetMetricFixedMask(mask)

        ## Choose Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)    # Linear interpolation is often sufficient
        
        ## Choose Optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, 
                                                        numberOfIterations=70, 
                                                        convergenceMinimumValue=1e-7, 
                                                        convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        ## For multi-level registration: the registration is performed on different resolution levels from rough to fine
        ## If not necessary, just comment it out
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        

        ## Execute registration
        registration_method.Execute(image_in1, image_in2)
        image_out = sitk.Resample(image_in2, image_in1, transform, 
                                        sitk.sitkLinear, -1.0, 
                                        image_in2.GetPixelIDValue())
        sitk.WriteImage(image_out, fnameOut)
    
