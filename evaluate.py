import numpy as np
import skimage.measure
from feature_matching import Projection_Preprocessing
import re
from skimage.metrics import structural_similarity,normalized_root_mse
import skimage.measure
import skimage.morphology
import cv2
import cal
import os
import SimpleITK as sitk
import scipy.ndimage
import i0_data
import load_data
import utils
import config

margin=0

def evalNeedleArea(img, img2, projname, name):
    if False and projname in ("genA", "201020"):
        if img.shape[0] > 500:
            pos = [408, 472-margin, 498-margin]
            mult = 2
        else:
            pos = [204, 236-margin, 249-margin]
            mult = 1
    elif projname in ("genB", "2010201"):
        if img.shape[0] > 500:
            pos = [465, 514-margin, 525-margin]
            mult = 2
        else:
            pos = [232, 257-margin, 262-margin]
            pos = [199, 258-margin, 257-margin]
            mult = 1
    elif projname in ("genC", ""):
        if img.shape[0] > 500:
            pos = [465, 514-margin, 525-margin]
            mult = 2
        else:
            #pos = [232, 257-margin, 262-margin]
            pos = [169, 158, 157]
            mult = 1
    else:
        return 0
    mask = np.zeros_like(img, dtype=bool)
    mask[pos[0],pos[1]-2:pos[1]+2,pos[2]-2:pos[2]+2] = True
    #rec = sitk.GetImageFromArray(1.0*mask)
    #rec.SetOrigin(out_rec_meta[0])
    #out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    #rec.SetSpacing(out_spacing)
    #sitk.WriteImage(rec, os.path.join("recos", "area_"+name+"_input1.nrrd"), True)
    #lines = img[mask].reshape((1,4,4))
    lines = img*mask
    maxpos = np.argmax(lines)
    maxpos = np.unravel_index(maxpos, lines.shape)
    maxval = lines[maxpos]
    
    mask = np.zeros_like(img, dtype=bool)
    mask[pos[0]-10:pos[0]+10] = True
    mask = skimage.morphology.binary_opening(mask*(img>(maxval*0.5)), skimage.morphology.cube(2))
    
    #rec = sitk.GetImageFromArray(mask*1.0)
    #rec.SetOrigin(out_rec_meta[0])
    #out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    #rec.SetSpacing(out_spacing)
    #sitk.WriteImage(rec, os.path.join("recos", "area_"+name+"_input2.nrrd"), True)

    labels = skimage.measure.label(mask)
    mask1 = labels==labels[maxpos]

    mask = np.zeros_like(img2, dtype=bool)
    mask[pos[0],pos[1]-2:pos[1]+2,pos[2]-2:pos[2]+2] = True
    lines = img2*mask
    maxpos = np.argmax(lines)
    maxpos = np.unravel_index(maxpos, lines.shape)
    maxval = lines[maxpos]
    mask = np.zeros_like(img2, dtype=bool)
    mask[pos[0]-10:pos[0]+10] = True
    mask = skimage.morphology.binary_opening(mask*(img2>(maxval*0.5)), skimage.morphology.cube(2))
    labels = skimage.measure.label(mask)
    mask2 = labels==labels[maxpos]

    im_sum = np.sum(mask1) + np.sum(mask2)
    intersection = np.logical_and(mask1, mask2)

    dice = 2. * np.sum(intersection) / im_sum


    #rec = sitk.GetImageFromArray(mask1*1.0)
    #rec.SetOrigin(out_rec_meta[0])
    #out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    #rec.SetSpacing(out_spacing)
    #sitk.WriteImage(rec, os.path.join("recos", "area_"+name+"_input1.nrrd"), True)

    #rec = sitk.GetImageFromArray(mask2*1.0)
    #rec.SetOrigin(out_rec_meta[0])
    #out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    #rec.SetSpacing(out_spacing)
    #sitk.WriteImage(rec, os.path.join("recos", "area_"+name+"_input2.nrrd"), True)
    #dice2 =  distance.dice(mask1, mask2)

    #print(dice, dice2)
    return dice

    #return np.count_nonzero(mask) * out_spacing[2] * out_spacing[2]

#def evalFWHM(img, name, pos = [241,261,265]):
def evalFWHM(img, name, pos = [232, 262, 257]):
    mult = 1
    if name in ("genA", "201020"):
        if img.shape[0] > 500:
            pos = [408, 472, 498]
            mult = 2
        else:
            pos = [204, 236, 249]
            mult = 1
    elif name in ("genB", "2010201"):
        if img.shape[0] > 500:
            pos = [465, 514, 525]
            mult = 2
        else:
            pos = [232, 257, 262]
            mult = 1
    else:
        return 0

    mask = np.zeros_like(img, dtype=bool)
    mask[pos[0],pos[1]-2:pos[1]+2,pos[2]-2:pos[2]+2] = True
    #rec = sitk.GetImageFromArray(img*mask)
    #rec.SetOrigin(out_rec_meta[0])
    #out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    #rec.SetSpacing(out_spacing)
    #sitk.WriteImage(rec, os.path.join("recos", "fwhm_"+name+"_input1.nrrd"), True)
    lines = img[mask].reshape((1,4,4))
    maxpos = np.argmax(lines)
    maxpos = np.unravel_index(maxpos, lines.shape)
    mask = np.zeros_like(img, dtype=bool)
    mask[maxpos[0]+pos[0],maxpos[1]+pos[1]-2,:] = True
    #rec = sitk.GetImageFromArray(img*mask)
    #rec.SetOrigin(out_rec_meta[0])
    out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    #rec.SetSpacing(out_spacing)
    #sitk.WriteImage(rec, os.path.join("recos", "fwhm_"+name+"_input2.nrrd"), True)
    line = img[mask]
    maxpos = maxpos[2]+pos[2]-2
    maxval = line[maxpos]
    i = maxpos
    left = 0
    right = len(line)-1
    while i > 0:
        if line[i] == 0.5*maxval:
            left = i
            break
        if line[i] < 0.5*maxval:
            left = i + (maxval-line[i]) / (line[i+1]-line[i])
            break
        i -= 1
    i = maxpos
    while i < len(line):
        if line[i] == 0.5*maxval:
            right = i
            break
        if line[i] < 0.5*maxval:
            right = i-1 + (maxval-line[i]) / (line[i-1]-line[i])
            break
        i += 1
    fwhm = (right-left) * out_spacing[2]
    print("FWHM: ", fwhm)
    return fwhm

def evalPerformance(output_in, real_in, runtime, name, stats_file='stats.csv', real_fwhm=None, real_area=None, gi_config={"GIoldold":[None], "absp1":[None], "p1":[None]}):
    if runtime == 0:
        output = output_in
        real = real_in
    if runtime == 1:
        output = Projection_Preprocessing(output_in)
        real = Projection_Preprocessing(real_in)
    if runtime == 2:
        output = (output_in-np.mean(output_in))/(np.std(output_in))
        real = (real_in-np.mean(real_in))/(np.std(real_in))
    if True or runtime == 3:
        output = (output_in-np.min(output_in))/(np.max(output_in)-np.min(output_in))
        real = (real_in-np.min(real_in))/(np.max(real_in)-np.min(real_in))
    if runtime == 4:
        output = (output_in-np.min(output_in))/(np.max(output_in)-np.min(output_in))
        output = (output-np.mean(output))/np.std(output)
        real = (real_in-np.min(real_in))/(np.max(real_in)-np.min(real_in))
        real = (real-np.mean(real))/np.std(real)
    if runtime == 5:
        output = (output_in-np.mean(output_in))/(np.max(output_in)-np.min(output_in))
        real = (real_in-np.mean(real_in))/(np.max(real_in)-np.min(real_in))
    #print(name, output.shape, real.shape)
    #name = name + str(runtime)


    #sitk.WriteImage(sitk.GetImageFromArray(output), name + "_out.nrrd")
    #sitk.WriteImage(sitk.GetImageFromArray(real), name + "_real.nrrd")

    if np.size(output[0]) != np.size(real[0]):
        print(name, np.size(output[0]), output.shape, np.size(real[0]), real.shape)
        return
    vals = []
    #for i in range(len(real)):
    if "rec" in stats_file:
        vals.append(1.0/cal.calcGIObjective(real, output, 0, None, {"GIoldold":[None], "absp1":[None], "p1":[None]}))
    else:
        for i in range(real.shape[0]):
            v = 1.0/cal.calcGIObjective(real[i], output[i], 0, None, {"GIoldold":[None], "absp1":[None], "p1":[None]})
            if not v!=v:
                vals.append(v)
    print("NGI: ", np.mean(vals), np.median(vals))

    vals1 = []
    #for i in range(len(real)):
    #    vals1.append(np.mean(cv2.matchTemplate(real[i], output[i], cv2.TM_CCORR_NORMED)))

    vals2 = []
    #for i in range(len(real)):
    #    vals2.append(np.mean(cv2.matchTemplate(real[i], output[i], cv2.TM_CCOEFF_NORMED)))

    vals3 = []
    vals4 = []
    for i in range(len(real)):
        break
        a = real[i].flatten()
        b = output[i].flatten()
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
        vals3.append(np.mean(cv2.matchTemplate(a, b, cv2.TM_CCORR)))
        vals4.append(np.mean(cv2.matchTemplate(a, b, cv2.TM_CCOEFF)))
        
    vals5 = []
    #for i in range(len(real)):
    #vals5.append(mutual_information_2d(real, output, normalized=True))

    vals6 = []
    #for i in range(len(real)):
    #print(real.dtype, output.dtype)
    if "rec" in stats_file:
        vals6.append(structural_similarity(real, output))
    else:
        for i in range(real.shape[0]):
            vals6.append(structural_similarity(real[i], output[i]))

    print("SSIM: ", np.mean(vals6), np.median(vals6))

    vals7 = []
    #for i in range(len(real)):
    if "rec" in stats_file:
        vals7.append(normalized_root_mse(real, output))
    else:
        for i in range(real.shape[0]):
            v = normalized_root_mse(real[i], output[i])
            if not v!=v:
                vals7.append(v)
    print("NRMSE: ", np.mean(vals7), np.median(vals7))

    vals8 = []
    #if "rec" in stats_file:
        #vals8.append(evalFWHM(output, "genB"))
        #if real_fwhm is None:
        #    real_fwhm = evalFWHM(real, "genB")
        #vals8.append(vals8[-1]-real_fwhm)
    
    vals9 = []
    if "rec" in stats_file and "proj" not in stats_file:
        vals9.append(evalNeedleArea(output, real, "genC", name))
        #if real_area is None:
        #    real_area = evalNeedleArea(real, "genB", name)
        #vals9.append(vals9[-1]-real_area)
        print("Area: ", vals9[-1])
    
    with open(stats_file, "a") as f:
        #f.write("{0};NGI;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals]) + "\n")
        #f.write("{0};CCORR_NORM;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals1]) + "\n")
        #f.write("{0};CCOEFF_NORM;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals2]) + "\n")
        #f.write("{0};NCCORR;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals3]) + "\n")
        #f.write("{0};NCCOEFF;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals4]) + "\n")
        #f.write("{0};NMI;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals5]) + "\n")
        #f.write("{0};SSIM;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals6]) + "\n")
        #f.write("{0};NRMSE;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals7]) + "\n")
        #if "rec" in stats_file:
        #    f.write("{0};FWHM;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals8]) + "\n")
        if "rec" in stats_file:
        #    f.write("{0};Area;{1};=MIN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 4, 1, {2}));=MAX(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 3, 1, {2}));=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=STDEV.P(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, runtime/(24*60*60), len(vals)) + ";".join([str(v) for v in vals9]) + "\n")
            print(stats_file, "bytes written", f.write(" & ".join([str(d) for d in [name, runtime/(24*60*60), np.mean(vals), np.mean(vals6), np.mean(vals7), np.mean(vals9)]]) + "\n"))
        else:
            print(name, len(vals), len(vals6), len(vals7))
            print(stats_file, "bytes written", f.write(" & ".join([str(d) for d in [name, runtime/(24*60*60), np.mean(vals), np.mean(vals6), np.mean(vals7)]]) + "\n"))

def evalSinoResults(out_path, in_path, projname, methods=None):
    ims, ims_un, mas, kvs, angles, coord_systems, sids, sods = load_data.read_dicoms(in_path)
    detector_shape = np.array((1920,2480))
    detector_mult = int(np.floor(detector_shape[0] / ims_un.shape[1]))
    detector_edges = int(detector_shape[0] / detector_mult - ims_un.shape[1]), int(detector_shape[1] / detector_mult - ims_un.shape[2])
    i0_ims, i0_mas, i0_kvs = i0_data.i0_data(detector_mult, ims_un.shape[1])
    res = np.mean( np.mean(i0_ims, axis=(1,2))[:,np.newaxis,np.newaxis] / i0_ims, axis=0)
    i0s = i0_data.i0_interpol(i0_ims, i0_mas, np.mean(mas))
    i0s[i0s==0] = 1e-8
    ims_un[ims_un==0] = 1e-8
    #print(detector_shape, detector_mult, detector_edges, ims_un.shape, i0_ims.shape, i0s.shape, np.count_nonzero(ims_un==0))
    ims_norm = (np.array(-np.log(ims_un/i0s), dtype=np.float32))
    #print(np.count_nonzero(~np.isfinite(ims_norm)),np.count_nonzero(~np.isfinite(ims_un)),np.count_nonzero(~np.isfinite(i0s)))
    #print(ims_norm.flatten()[np.isnan(ims_norm).flatten()][:10], ims_un.flatten()[np.isnan(ims_norm).flatten()][:10], i0s.flatten()[np.isnan(np.sum(ims_norm,axis=0)).flatten()][:10])
    ims_norm[np.isnan(ims_norm)] = 0

    print("forcast_"+projname+"4_sino-output", out_path)
    for filename in os.listdir(out_path):
        #if re.fullmatch("forcast_"+projname+"4_sino-output", filename) != None:
        if "forcast_"+projname+"4_sino-output" in filename:
        #if re.fullmatch("target_sino.nrrd", filename) != None:
            img = sitk.ReadImage(os.path.join(out_path, filename))
            proj = (sitk.GetArrayFromImage(img))
            if proj.dtype != np.float32:
                proj = np.array(proj, dtype=np.float32)
            ims_norm2 = np.swapaxes(proj, 0, 1)

    def sino_data():
        input_sino = True
        for filename in os.listdir(out_path):
            if re.fullmatch("forcast_(.+?)_sino-output.nrrd", filename) != None and projname in filename and float(filename.split("_")[4]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = (sitk.GetArrayFromImage(img))
                if proj.dtype != np.float32:
                    proj = np.array(proj, dtype=np.float32)
                img = sitk.ReadImage(os.path.join(out_path, filename.replace("sino-output", "projs-input")))
                target = (sitk.GetArrayFromImage(img))
                if target.dtype != np.float32:
                    target = np.array(target, dtype=np.float32)
                #projs.append(proj)
                #names.append("_".join(filename.split('_')[1:-1]))
                yield "_".join(filename.split('_')[1:-1]), proj, target
            if re.fullmatch("forcast_(.+?)_sino-input.nrrd", filename) != None and projname in filename and input_sino == False:
                input_sino = True
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = (sitk.GetArrayFromImage(img))
                if proj.dtype != np.float32:
                    proj = np.array(proj, dtype=np.float32)
                #projs.append(proj)
                #names.append(projname+"_input")
                yield projname+"_input", proj
            if re.fullmatch("trajtomo_reco_matlab_(.+?)_output.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = (sitk.GetArrayFromImage(img))
                if proj.dtype != np.float32:
                    proj = np.array(proj, dtype=np.float32)
                #projs.append(proj)
                #names.append("_".join(filename.split('_')[3:-1])+"_matlab")
                yield "_".join(filename.split('_')[3:-1])+"_matlab", proj

    for name, proj, target in sino_data():
        print(name)
        skip = int(np.ceil(ims_un.shape[0]/proj.shape[1]))

        #i0s = [i0_est(ims_un[::skip][i], proj[:,i]) for i in range(proj.shape[1])]
        #ims = np.array(-np.log(ims_un[::skip]/np.mean(i0s)), dtype=np.float32)

        #i0s = i0_interpol(i0_ims, i0_mas, np.mean(mas))
        #ims = -np.log(ims/i0s)[::skip]
        ims = ims_norm[::skip]
        ims2 = ims_norm2[::skip]

        i0s = np.array([i0_data.i0_est(ims_un[i], proj[:,i])*res for i in range(ims_un.shape[0])])
        i0s = np.mean(i0s, axis=0)
        i0s[i0s==0] = 1e-8
        ims_norm3 = (np.array(-np.log(ims_un/i0s), dtype=np.float32))
        ims3 = ims_norm3[::skip]

        try:
            #evalPerformance(np.swapaxes(proj, 0, 1), ims, 0, name, 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims2, 0, name+"sim", 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims, 1, name, 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims2, 1, name+"sim", 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims, 2, name, 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims2, 2, name+"sim", 'stats_proj.csv')
            evalPerformance(np.swapaxes(proj, 0, 1), ims, 0, name, 'stats_proj.csv')
            evalPerformance(np.swapaxes(proj, 0, 1), ims2, 0, name+"sim", 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims, 0, name, 'stats_proj_rec.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims2, 0, name+"sim", 'stats_proj_rec.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims, 4, name, 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims2, 4, name+"sim", 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims, 5, name, 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims2, 5, name+"sim", 'stats_proj.csv')
            #evalPerformance(np.swapaxes(proj, 0, 1), ims3, 0, name+"i0", 'stats_proj.csv')
            evalPerformance(np.swapaxes(proj, 0, 1), np.swapaxes(target,0,1), 0, name+"proj", 'stats_proj.csv')
            #evalPerformance(Projection_Preprocessing(np.swapaxes(proj, 0, 1)), Projection_Preprocessing(ims), 0, name+"pre", 'stats_proj.csv')
        except Exception as e:
            print(e)
            raise e
    
def evalRecoResults(out_path, in_path, projname, methods=None):
    global out_rec_meta
    input_recos = {}
    input_gi_confs = {}
    input_fwhm = {}
    input_area = {}
    metas = {}
    for filename in os.listdir(out_path):
        #if re.fullmatch("forcast_[^_]+_reco-input.nrrd", filename) != None:
        if re.fullmatch("forcast_201020(1_imbu|_imbureg_noimbu|_imbu)_cbct_4_reco-output.nrrd", filename) != None:
        #if re.fullmatch("target_reco.nrrd", filename) != None:
            try:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                real_img = np.array(sitk.GetArrayFromImage(img), dtype=np.float32)
                #print(real_img.shape, end=" -> ")
                #real_img = scipy.ndimage.zoom(real_img, 0.5, order=1)
                #print(real_img.shape)
                real_name = filename.split('_')[1]
                #real_name = ""
                if False and real_name == '201020':
                    continue
                #real_name = "genB"
                #real_name = real_name.replace('2010201', '201020')
                metas[real_name] = (img.GetOrigin(), img.GetSize(), img.GetSpacing(), real_img.shape)
                input_recos[real_name] = real_img
                input_gi_confs[real_name] = {"GIoldold":[None], "absp1":[None], "p1":[None]}
                out_rec_meta = metas[real_name]
                print(real_name, filename, real_img.shape)
                input_fwhm[real_name] = evalFWHM(real_img, real_name)
                input_area[real_name] = evalNeedleArea(real_img, real_img, real_name, real_name)
                if False and '2010201' in real_name:
                    oname = real_name.replace('2010201', '201020')
                    metas[oname] = metas[real_name]
                    input_recos[oname] = real_img
                    input_gi_confs[oname] = input_gi_confs[real_name]
                    out_rec_meta = metas[real_name]
                    input_fwhm[oname] = input_fwhm[real_name]
                    input_area[oname] = input_area[real_name]

                evalPerformance(real_img[:,margin:-margin,margin:-margin], real_img[:,margin:-margin,margin:-margin], 0, real_name, 'stats_rec.csv', real_fwhm=input_fwhm[real_name], real_area=input_area[real_name], gi_config=input_gi_confs[real_name])
            except Exception as e:
                print(e)
        if re.fullmatch("forcast_201020_imbu_cbct_4_reco-output.nrrd", filename) != None:
            img = sitk.ReadImage(os.path.join(out_path, filename))
            real_img = np.array(sitk.GetArrayFromImage(img), dtype=np.float32)
            input_recos["2010202"] = real_img


    def reco_data():
        input_sino = False
        for filename in os.listdir(out_path):
            if re.fullmatch("forcast_matlab_(.+?)_reco-output.nrrd", filename) != None and projname in filename and float(filename.split("_")[-2]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[2:]))
                yield "_".join(filename.split('_')[2:]), proj, filename
            elif re.fullmatch("forcast_matlab_(.+?)_reco-output_sirt.nrrd", filename) != None and projname in filename and float(filename.split("_")[-3]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[2:]))
                yield "_".join(filename.split('_')[2:]), proj, filename
            elif re.fullmatch("forcast_matlab_(.+?)_reco-output_cgls.nrrd", filename) != None and projname in filename and float(filename.split("_")[-3]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[2:]))
                yield "_".join(filename.split('_')[2:]), proj, filename
            elif re.fullmatch("forcast_(.+?)_reco-output_sirt.nrrd", filename) != None and projname in filename and float(filename.split("_")[-3]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[1:]))
                yield "_".join(filename.split('_')[1:]), proj, filename
            elif re.fullmatch("forcast_(.+?)_reco-output_cgls.nrrd", filename) != None and projname in filename and float(filename.split("_")[-3]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[1:]))
                yield "_".join(filename.split('_')[1:]), proj, filename
            elif re.fullmatch("forcast_(.+?)_reco-output.nrrd", filename) != None and projname in filename and float(filename.split("_")[-2]) in methods:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[1:]))
                yield "_".join(filename.split('_')[1:]), proj, filename
            elif re.fullmatch("trajtomo_reco_matlab_"+projname+"_reg\.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[3:])+"_matlab")
                yield "_".join(filename.split('_')[3:])+"_matlab", proj, filename
            elif re.fullmatch("forcast_"+projname.split("_")[0]+"_reco-input.nrrd", filename) != None:# and projname in filename and input_sino == False:
                continue
                input_sino = True
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #names.append(projname+"_input")
                yield projname+"_input", proj, filename
            #elif re.fullmatch("forcast_[^_]+_reco-input.nrrd", filename) != None:
            #    img = sitk.ReadImage(os.path.join(out_path, filename))
            #    input_recos[filename.split('_')[1]] = sitk.GetArrayFromImage(img)

    for name, proj, filename in reco_data():
        print(filename, name)
        real_name = name.split('_')[0].replace('A', 'B')
        #real_name = "2010201"
        ims = np.array(input_recos[real_name])
        #if (np.array(proj.shape)!=np.array(ims.shape)).any():
        #    ims = np.array(scipy.ndimage.zoom(ims, np.array(proj.shape)/np.array(ims.shape), order=1), dtype=np.float32)
        #    proj = np.array(proj, dtype=np.float32)
        if (np.array(proj.shape)!=np.array(ims.shape)).any():
            proj = np.array(scipy.ndimage.zoom(proj, np.array(ims.shape)/np.array(proj.shape), order=1), dtype=ims.dtype)
        #if "input" not in name:
        #    continue
        if ims.shape == proj.shape:
            out_rec_meta = metas[real_name]
            try:
                mask = utils.create_circular_mask(proj.shape, radius_off=margin)
                #write_images(1.0*mask, "mask.nrrd")
                output = (proj*mask)[30:-30,margin:-margin,margin:-margin]
                real = (ims*mask)[30:-30,margin:-margin,margin:-margin]
                #print(output.shape, real.shape)
                input_gi_confs[real_name] = {"GIoldold":[None], "absp1":[None], "p1":[None]}
                #evalPerformance(output, real, 0, name, 'stats_rec.csv', real_fwhm=input_fwhm[real_name], real_area=input_area[real_name], gi_config=input_gi_confs[real_name])
                evalPerformance(Projection_Preprocessing(output), Projection_Preprocessing(real), 0, name+"pre", 'stats_rec.csv', real_fwhm=input_fwhm[real_name], real_area=input_area[real_name], gi_config=input_gi_confs[real_name])

                #if "_0_" in name:
                #    diff = proj-input_recos["201020"]
                #    write_images(-diff, os.path.join(out_path, "diff_" + name))
                #    diff = proj-input_recos["2010201"]
                #    write_images(-diff, os.path.join(out_path, "diff2_" + name))
                #    diff = proj-input_recos["2010202"]
                #    write_images(-diff, os.path.join(out_path, "diff3_" + name))
                #else:
                #    diff = proj-input_recos["201020"]
                #    write_images(diff, os.path.join(out_path, "diff_" + name))
                #    diff = proj-input_recos["2010201"]
                #    write_images(diff, os.path.join(out_path, "diff2_" + name))
                #    diff = proj-input_recos["2010202"]
                #    write_images(diff, os.path.join(out_path, "diff3_" + name))
            except Exception as ex:
                print(name, "failed", ex)
                raise

def evalAllResults(evalSino=True, evalReco=True, outpath="recos"):
    projs = config.get_proj_paths()
    if evalSino:
        with open("stats_proj.csv", "w") as f:
            f.truncate()
            f.write(" & ".join(["Name", "Runtime", "NGI", "SSIM", "NRMSE"]) + "\\\\\n")
        with open("stats_proj_rec.csv", "w") as f:
            f.truncate()
            f.write(" & ".join(["Name", "Runtime", "NGI", "SSIM", "NRMSE", "Dice"]) + "\\\\\n")
        for name, proj_path, _, methods in projs:
            print(proj_path)
            evalSinoResults(outpath, proj_path, name, methods)
    if evalReco:        
        with open("stats_rec.csv", "w") as f:
            f.truncate()
            f.write(" & ".join(["Name", "Runtime", "NGI", "SSIM", "NRMSE", "Area"]) + "\\\\\n")
        for name, proj_path, _, methods in projs:
            print(proj_path)
            evalRecoResults(outpath, proj_path, name, methods)
