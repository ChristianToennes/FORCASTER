import pydicom
import os
import os.path
import SimpleITK as sitk

indir = 'input\\CKM_LUMBALSPINE_20_10_13-09_13_33-DST-1_3_12_2_1107_5_4_5_160969'
outdir = 'output'

filepaths = set()

for root, dirs, files in os.walk(indir):
    for entry in files:
        path = os.path.abspath(os.path.join(root, entry))
        #read DICOM files
        ds = pydicom.dcmread(path)
        try:
            new_path = os.path.join(outdir, str(ds.PatientName), ds.StudyDate+'-'+ds.StudyTime, ds.SeriesDescription)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            filepaths.add((str(ds.PatientName), ds.StudyDate+"-"+ds.StudyTime, ds.SeriesDescription))
            new_file_path = new_path + "\\" + ds.SeriesDate+'-'+ds.SeriesTime + "." + str(ds.InstanceNumber).zfill(5)+'.dcm'
            #files[new_path].append(new_file_path)
            if not os.path.exists(new_file_path):
                ds.save_as(new_file_path)
        except Exception as e:
            print(str(e), path, dir(ds))

for (name, time, desc) in filepaths:
    try:
        reader = sitk.ImageSeriesReader()
        path = os.path.join(outdir, name, time, desc)
        print(path)
        filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path, "", True, True, True)
        reader.SetFileNames(filenames)
        reader.SetOutputPixelType(sitk.sitkFloat32)
        #reader.SetFileNames(list([fnameFixed + "/" + str(f) + ".dcm" for f in reversed(sorted([int(f.split(".")[0]) for f in os.listdir(fnameFixed)]))]))
        image = reader.Execute()
        new_path = os.path.join(outdir, name, time)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        sitk.WriteImage(image, new_path + "\\" + desc + ".nrrd")
    except Exception as e:
        print(e)