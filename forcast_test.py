import numpy as np
import SimpleITK as sitk
#import forcast
import cal
import utils
#import i0
import os
import pydicom
import struct
#import scipy.interpolate
import astra
import time
import itertools
import cProfile, pstats
import cv2
import scipy.ndimage
import re

def write_vectors(name, vecs):
    return
    with open(name+".csv", "w") as f:
        f.writelines([",".join([str(v) for v in vec])+"\n" for vec in vecs])

def read_vectors(path):
    with open(path, "r") as f:
        res = np.array([[float(v) for v in l.split(',')] for l in f.readlines().split()])
    return res

def smooth(vecs):
    res = np.zeros_like(vecs)
    from scipy.signal import savgol_filter
    for i in range(vecs.shape[1]):
        res[:,i] = savgol_filter(vecs[:,i], 11, 3) # window size 51, polynomial order 3
    return res

def normalize(images, mAs_array, kV_array, percent_gain):
    print("normalize images")
    kVs = {}
    kVs[40] = (20.4125, 677.964)
    kVs[50] = (61.6163, 686.4824)
    kVs[60] = (138.4021, 684.1844)
    kVs[70] = (250.8008, 691.9573)
    kVs[80] = (398.963, 701.1038)
    kVs[90] = (586.5949, 711.416)
    kVs[100] = (794.5124, 729.8813)
    kVs[109] = (1006.1, 750.0054)
    kVs[120] = (1252.2, 791.9865)
    kVs[125] = (1404.2202, 796.101)

    #f, gamma = i0.get_i0(prefix + r"\70kVp")
    #kVs[70] = f

    fs = []
    gain = 3
    #gain = 1
    for mAs, kV in zip(mAs_array, kV_array):
        if kV in kVs:
            f = np.polyval(kVs[kV], mAs)
        else:
            kVs_keys = np.array(list(sorted(kVs.keys())))
            if kV < kVs_keys[0]:
                f = np.polyval(kVs[kVs_keys[0]], mAs)
            elif kV > kVs_keys[-1]:
                f = np.polyval(kVs[kVs_keys[-1]], mAs)
            else:
                i1, i2 = np.argsort(np.abs(kVs_keys-kV))[:2]
                f1 = np.polyval(kVs[kVs_keys[i1]], mAs)
                f2 = np.polyval(kVs[kVs_keys[i2]], mAs)
                d1 = np.abs(kVs_keys[i1]-kV)*1.0
                d2 = np.abs(kVs_keys[i2]-kV)*1.0
                f = f1*(1.0-(d1/(d1+d2))) + f2*(1.0-(d2/(d1+d2)))
        fs.append(f)
    
    fs = np.array(fs).flatten()

    skip = 4
    if images.shape[1] < 1000:
        skip = 2
    #if images.shape[1] < 600:
    #    skip = 1

    edges = 1

    sel = np.zeros(images.shape, dtype=bool)
    #sel[:,20*skip:-20*skip:skip,20*skip:-20*skip:skip] = True
    #norm_images_gained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    #norm_images_ungained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    sel[:,edges:-edges:skip,edges:-edges:skip] = True
    sel_shape = (images.shape[0],images[0,edges:-edges:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,edges:-edges:skip].shape[0])))
    norm_images_gained = np.zeros(sel_shape, dtype=np.float32)
    norm_images_ungained = np.zeros(sel_shape, dtype=np.float32)
    
    gained_images = np.array([image*(1+gain/100) for image,gain in zip(images[sel].reshape(sel_shape), percent_gain)])
    
    for i in range(len(fs)):
        norm_img = gained_images[i].reshape(norm_images_gained[0].shape)
        norm_images_gained[i] = norm_img
        norm_img = images[i][sel[i]].reshape(norm_images_gained[0].shape)
        norm_images_ungained[i] = norm_img

    gain = 2.30
    #gain = 1
    offset = 400
    use = fs>1
    while (np.max(norm_images_gained, axis=(1,2))[use] > (gain*fs[use])).all():
        gain += 1
    #gain = np.ones_like(fs)
    #offset = 0
    for i in range(len(fs)):
        norm_img = norm_images_gained[i] / (offset + (gain*fs)[i])
        #norm_img = norm_images_gained[i] / (1.1*np.max(norm_images_gained[i]))
        if (norm_img==0).any():
            norm_img[norm_img==0] = np.min(norm_img[norm_img!=0])
        norm_images_gained[i] = -np.log(norm_img)

    return norm_images_gained, norm_images_ungained, offset+gain*fs, fs

def read_dicoms(indir, max_ims=np.inf):
    print("read dicoms")
    kvs = []
    mas = []
    μas = []
    ts = []
    thetas = []
    phis = []
    prims = []
    secs = []
    ims = []
    percent_gain = []
    coord_systems = []
    sids = []
    sods = []

    #sid = []
    for root, _dirs, files in os.walk(indir):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            #read DICOM files
            ds = pydicom.dcmread(path)
            if "PositionerPrimaryAngleIncrement" in dir(ds):
                ims = ds.pixel_array
                ts = list(range(len(ims)))
                thetas = np.array(ds.PositionerPrimaryAngleIncrement)
                phis = np.array(ds.PositionerSecondaryAngleIncrement)
                
                if ds[0x0021,0x1059].VR == "FL":
                    cs = np.array(ds[0x0021,0x1059].value).reshape((len(ts), 3, 4))
                    coord_systems = np.array(ds[0x0021,0x1059].value).reshape((len(ts), 3, 4))
                else:
                    cs = np.array(list(struct.iter_unpack("<f", ds[0x0021,0x1059].value))).reshape((len(ts), 3, 4))
                    coord_systems = np.array(list(struct.iter_unpack("<f", ds[0x0021,0x1059].value))).reshape((len(ts), 3, 4))
                
                if ds[0x0021,0x1031].VR == "SS":
                    sids = np.array(ds[0x0021,0x1031].value)*0.1
                else:
                    sids = np.array(list(struct.iter_unpack("<h", ds[0x0021,0x1031].value))).flatten()*0.1
                
                
                #if ds[0x0021,0x1017].VR == "SL":
                #    sods = [np.array(ds[0x0021,0x1017].value)]*len(ts)
                #else:
                #    sods = [np.array(int.from_bytes(ds[0x0021,0x1017].value, "little", signed=True))]*len(ts)
                
                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)
                sods = [stparmdata["SOD_A"]] *len(ts)

                if ds[0x0019,0x1008].VR == "US":
                    percent_gain = [ds[0x0019,0x1008].value] * len(ts)
                else:
                    percent_gain = [float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False))] * len(ts)

                #sid = ds[0x0021,0x1031].value
                if ds[0x0021,0x100f].VR =="SL":
                    xray_info = np.array(ds[0x0021,0x100F].value)
                else:
                    xray_info = np.array(list(struct.iter_unpack("<l",ds[0x0021,0x100f].value))).flatten()
                kvs = xray_info[0::4]
                mas = xray_info[1::4]*xray_info[2::4]*0.001
                μas = xray_info[2::4]*0.001

            elif "NumberOfFrames" in dir(ds):
                ims = ds.pixel_array
                for i in range(int(ds.NumberOfFrames)):
                    ts.append(len(ts))
                    kvs.append(float(ds.KVP))
                    mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                    if ds[0x0021,0x1004].VR == "SL":
                        μas.append(float(ds[0x0021,0x1004].value)*0.001)
                    else:
                        μas.append(struct.unpack("<l", ds[0x0021,0x1004].value)[0]*0.001)
                    thetas.append(float(ds.PositionerPrimaryAngle))
                    phis.append(float(ds.PositionerSecondaryAngle))

                    stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                    cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))
                    cs[:,2] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,2])
                    cs[:,1] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,1])

                    coord_systems.append(cs)
                    sids.append(np.array(stparmdata["SID_A"]))
                    sods.append(np.array(stparmdata["SOD_A"]))

                    rv = cs[:,2]
                    rv /= np.sum(rv**2, axis=-1)
                    prim = np.arctan2(np.sqrt(rv[0]**2+rv[1]**2), rv[2])
                    prim = np.arccos(rv[2] / np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2) )
                    prim = prim*180 / np.pi
                    

                    if cs[1,2]<0:
                        prim *= -1

                    if (prim > 50 and prim < 135) or (prim <-45 and prim > -135):
                        sec = np.arctan2(rv[1], rv[0])
                        sec = sec * 180 / np.pi
                        if cs[1,2]<0:
                            sec *= -1
                        sec -= 90
                    else:
                        sec = np.arctan2(rv[2], rv[0])
                        sec = sec * 180 / np.pi - 90

                    prims.append(prim)
                    secs.append(sec)
                    if ds[0x0019,0x1008].VR == "US":
                        percent_gain.append(ds[0x0019,0x1008].value)
                    else:
                        percent_gain.append(float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False)))

            elif "PositionerPrimaryAngle" in dir(ds):
                ts.append(len(ts))
                kvs.append(float(ds.KVP))
                mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                if ds[0x0021,0x1004].VR == "SL":
                    μas.append(float(ds[0x0021,0x1004].value)*0.001)
                else:
                    μas.append(struct.unpack("<l", ds[0x0021,0x1004].value)[0]*0.001)
                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))
                ims.append(ds.pixel_array)

                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))

                coord_systems.append(cs)
                sids.append(np.array(stparmdata["SID_A"]))
                sods.append(np.array(stparmdata["SOD_A"]))

                rv = cs[:,2]
                rv /= np.sum(rv**2, axis=-1)
                prim = np.arctan2(np.sqrt(rv[0]**2+rv[1]**2), rv[2])
                prim = np.arccos(rv[2] / np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2) )
                prim = prim*180 / np.pi

                if cs[1,2]<0:
                    prim *= -1

                if (prim > 50 and prim < 135) or (prim <-45 and prim > -135):
                    sec = np.arctan2(rv[1], rv[0])
                    sec = sec * 180 / np.pi
                    if cs[1,2]<0:
                        sec *= -1
                    sec -= 90
                else:
                    sec = np.arctan2(rv[2], rv[0])
                    sec = sec * 180 / np.pi - 90

                prims.append(prim)
                secs.append(sec)

                if ds[0x0019,0x1008].VR == "US":
                    percent_gain.append(ds[0x0019,0x1008].value)
                else:
                    percent_gain.append(float(int.from_bytes(ds[0x0019,0x1008].value, "little", signed=False)))
            
            del ds
            if len(ts)>=max_ims:
                break
        if len(ts)>=max_ims:
            break

    print("create numpy arrays")
    kvs = np.array(kvs)
    mas = np.array(mas)
    μas = np.array(μas)
    percent_gain = np.array(percent_gain)
    thetas = np.array(thetas)
    phis = np.array(phis)
    ims = np.array(ims)
    coord_systems = np.array(coord_systems)
    sids = np.array(sids)
    sods = np.array(sods)

    cs = coord_systems
    
    rv = cs[:,:,2]
    rv /= np.vstack((np.sum(rv, axis=-1),np.sum(rv, axis=-1),np.sum(rv, axis=-1))).T
    prim = np.arctan2(np.sqrt(rv[:,0]**2+rv[:,1]**2), rv[:,2])
    prim = np.arccos(rv[:,2] / np.sqrt(rv[:,0]**2+rv[:,1]**2+rv[:,2]**2) )
    sec = 0.5*np.pi-np.arctan2(rv[:,1], rv[:,0])
    prim = prim*180 / np.pi
    sec = sec * 180 / np.pi

    prim[cs[:,1,2]<0] *= -1
    sec[cs[:,1,2]<0] -= 180
    prim[np.bitwise_and(cs[:,2,2]<0, cs[:,1,1]>0) ] -= 180
    prim[cs[:,1,0]>0] -= 180

    #thetas = prim
    #phis = sec
    angles = np.vstack((thetas*np.pi/180.0, phis*np.pi/180.0, np.zeros_like(thetas))).T

    ims_gained, ims_ungained, i0s_gained, i0s_ungained = normalize(ims, μas, kvs, percent_gain)

    return ims_gained, ims_ungained, i0s_gained, i0s_ungained, angles, coord_systems, sids, sods

def reg_rough(ims, params, config, c=0):
    corrs = [None]*len(params)
    noise = config["noise"]
    for i in reversed(range(len(params))):
    #for i in [29]:
        print(i, end=",", flush=True)
        cur = params[i]
        real_img = cal.Projection_Preprocessing(ims[i])
        config["real_img"] = real_img
        config["noise"] = (noise[0][i], noise[1][i])
        for si in range(1):
            try:
                old_cur = np.array(cur)
                cur = cal.roughRegistration(cur, config, c)
                noise[0][i], noise[1][i] = config["noise"]
            except Exception as ex:
                print(i, ex, cur)
                raise
            #if (np.abs(old_cur-cur)<1e-8).all():
            #    print(si, end=" ", flush=True)
            #    break
        corrs[i] = cur
        #print(flush=True)
        
    corrs = np.array(corrs)
    config["noise"] = noise
    #print(corrs)
    return corrs

import multiprocessing as mp
import sys
import io

def it_func(con, Ax_params, ready, name):
    if name != None:
        profiler = cProfile.Profile()
    try:
        
        #print("start")
        np.seterr(all='raise')
        Ax = None
        Ax = utils.Ax_param_asta(*Ax_params)
        while True:
            try:
                con.send(("ready",))
                ready.set()
                (i, cur, im, noise, method) = con.recv()
                #print(i)
                old_stdout = sys.stdout
                sys.stdout = stringout = io.StringIO()

                if name != None:
                    profiler.enable()    
                real_img = cal.Projection_Preprocessing(im)
                cur_config = {"real_img": real_img, "Ax": Ax, "noise": noise}
                try:
                    cur = cal.roughRegistration(cur, cur_config, method)
                except Exception as ex:
                    print(ex, i, cur, file=sys.stderr)
                #corrs.append(cur)
                if name != None:
                    profiler.disable()
                    profiler.dump_stats(name)
                stringout.flush()
                con.send(("result",i,cur,stringout.getvalue()))
                ready.set()
                stringout.close()
                sys.stdout = old_stdout
            except EOFError:
                break
            except BrokenPipeError:
                if name != None:
                    profiler.dump_stats(name)
                return
        if name != None:
            profiler.dump_stats(name)
        try:
            con.send(("error",))
        except EOFError:
            pass
        except BrokenPipeError:
            pass
    except KeyboardInterrupt:
        pass

def reg_rough_parallel(ims, params, config, c=0):
    corrs = []
    pool_size = mp.cpu_count()
    pool = []
    proc_count = 0
    #corrsq = mp.Queue()
    ready = mp.Event()
    #for _ in range(pool_size):
    #    cons.append(mp.Pipe(True))
    #    pool.append(mp.Process(target=it_func, args=(cons[-1][1], config["Ax_gen"], ready)))
    #    pool[-1].start()
    
    corrs = np.array([None]*len(params))
    indices = list(range(len(params)))

    while np.array([e is None for e in corrs]).any(): #len(indices)>0:
        ready_con = None
        while ready_con is None:
            for _ in range(len(pool), pool_size):
                p = mp.Pipe(True)
                if "profile" in config and config["profile"]:
                    name = config["name"]+"_"+str(proc_count)
                else:
                    name = None
                proc = mp.Process(target=it_func, args=(p[1], config["Ax_gen"], ready, name), daemon=True)
                proc.start()
                proc_count += 1
                pool.append([p[0], p[1], proc, -1])
            ready.clear()
            finished_con = []
            for con in pool:
                try:
                    if con[2].is_alive():
                        if con[0].poll():
                            res = con[0].recv()
                            if res[0] == "ready":
                                ready_con = con
                                break
                            elif res[0] == "result":
                                corrs[res[1]] = res[2]
                                #print(res[1], res[3], flush=True)
                                print(res[1], end='; ', flush=True)
                            elif res[0] == "error":
                                finished_con.append(con)
                            else:
                                print("error", res)
                    else:
                        finished_con.append(con)
                except (OSError, BrokenPipeError, EOFError):
                    finished_con.append(con)

            for con in finished_con:
                indices.append(con[3])
                pool.remove(con)
                con[0].close()
                con[1].close()
            if ready_con is None:
                ready.wait(1)
        if len(indices) > 0:
            i = indices.pop()
            ready_con[0].send((i, params[i], ims[i], (config["noise"][0][i],config["noise"][1][i]), c))
            ready_con[3] = i

    if False:
        while len(pool) > 0:
            finished_con = []
            ready.clear()
            for con in pool:
                try:
                    if con[2].is_alive():
                        if con[0].poll():
                            res = con[0].recv()
                            if res[0] == "ready" or res[0] == "error":
                                finished_con.append(con)
                            elif res[0] == "result":
                                corrs[res[1]] = res[2]
                                #print(res[1], res[3])
                                print(res[1], end=', ')
                    else:
                        finished_con.append(con)
                except (OSError, BrokenPipeError, EOFError):
                        finished_con.append(con)
            for con in finished_con:
                pool.remove(con)
                con[0].close()
                con[1].close()
            if len(pool) > 0:
                ready.wait(1)

    for con in pool:
        con[2].terminate()
        con[0].close()
        con[1].close()
        
    corrs = np.array(corrs.tolist())
    print()
    #print(corrs)
    return corrs

def create_circular_mask(shape, center=None, radius=None, radius_off=5, end_off=30):
    l, h, w = shape
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.zeros((l,h,w), dtype=bool)
    mask[:] = (dist_from_center <= (radius-radius_off))[np.newaxis,:,:]
    for i in range(30):
        mask[i] = (dist_from_center <= (i/30)*(radius-radius_off))
        mask[-i-1] = (dist_from_center <= (i/30)*(radius-radius_off))

    return mask

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    def print_val(val):
        if np.abs(val) > 0.1:
            print("{}{: .3f}{}".format(bcolors.RED, val, bcolors.END), end=", ")
        elif np.abs(val) < 0.1:
            print("{}{: .3f}{}".format(bcolors.GREEN, val, bcolors.END), end=", ")
        else:
            print("{}{: .3f}{}".format(bcolors.YELLOW, val, bcolors.END), end=", ")


def print_stats(noise):
    for axis in [0,1,2]:
        print("{}{}{}".format(bcolors.BLUE, axis, bcolors.END), end=": ")
        bcolors.print_val(np.mean(noise[:, axis]))
        bcolors.print_val(np.std(noise[:, axis]))
        bcolors.print_val(np.min(noise[:, axis]))
        bcolors.print_val(np.quantile(noise[:, axis], 0.25))
        bcolors.print_val(np.median(noise[:, axis]))
        bcolors.print_val(np.quantile(noise[:, axis], 0.75))
        bcolors.print_val(np.max(noise[:, axis]))
        print()

EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)



    jh = np.histogram2d(x.flatten(), y.flatten(), bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    scipy.ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi

def evalPerformance(output, real, time, name, stats_file='stats.csv'):
    vals = []
    for i in range(len(real)):
        vals.append(cal.calcGIObjective(output[:,i], real[i], {}))
    print("NGI: ", np.mean(vals))

    vals1 = []
    for i in range(len(real)):
        vals1.append(np.mean(cv2.matchTemplate(real[i], output[:,i], cv2.TM_CCORR_NORMED)))

    vals2 = []
    for i in range(len(real)):
        vals2.append(np.mean(cv2.matchTemplate(real[i], output[:,i], cv2.TM_CCOEFF_NORMED)))

    vals3 = []
    for i in range(len(real)):
        vals3.append(np.mean(cv2.matchTemplate(real[i], output[:,i], cv2.TM_CCOEFF)))

    vals4 = []
    for i in range(len(real)):
        vals4.append(np.mean(cv2.matchTemplate(real[i], output[:,i], cv2.TM_CCORR)))

    vals5 = []
    for i in range(len(real)):
        vals5.append(mutual_information_2d(real[i], output[:,i], normalized=True))

    with open(stats_file, "a") as f:
        f.write("{0};NGI;{1};=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, time/(24*60*60), len(vals)) + ";".join([str(v) for v in vals]) + "\n")
        f.write("{0};CCORR_NORM;{1};=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, time/(24*60*60), len(vals)) + ";".join([str(v) for v in vals1]) + "\n")
        f.write("{0};CCOEFF_NORM;{1};=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, time/(24*60*60), len(vals)) + ";".join([str(v) for v in vals2]) + "\n")
        f.write("{0};CCOEFF;{1};=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, time/(24*60*60), len(vals)) + ";".join([str(v) for v in vals3]) + "\n")
        f.write("{0};CCORR;{1};=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, time/(24*60*60), len(vals)) + ";".join([str(v) for v in vals4]) + "\n")
        f.write("{0};NMI;{1};=AVERAGE(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 2, 1, {2}));=MEDIAN(OFFSET(INDIRECT(ADDRESS(ROW(),COLUMN())), 0, 1, 1, {2}));".format(name, time/(24*60*60), len(vals)) + ";".join([str(v) for v in vals5]) + "\n")

def evalResults(out_path, in_path, projname):
    ims_un = read_dicoms(in_path)[1]

    projs = []
    names = []
    input_sino = False
    for filename in os.listdir(out_path):
        if re.fullmatch("forcast_(.+?)_sino-output.nrrd", filename) != None and projname in filename:
            img = sitk.ReadImage(os.path.join(out_path, filename))
            projs.append(sitk.GetArrayFromImage(img))
            names.append("_".join(filename.split('_')[1:-1]))
        if re.fullmatch("forcast_(.+?)_sino-input.nrrd", filename) != None and projname in filename and input_sino == False:
            input_sino = True
            img = sitk.ReadImage(os.path.join(out_path, filename))
            projs.append(sitk.GetArrayFromImage(img))
            names.append(projname+"_input")

    for name, proj in zip(names, projs):
        skip = int(np.ceil(ims_un.shape[0]/proj.shape[1]))
        
        i0s = [i0_est(ims_un[i::skip], proj[:,i]) for i in range(proj.shape[1])]
        ims = -np.log(ims_un[::skip]/np.mean(i0s))
        
        evalPerformance(proj, ims, 0, name, 'stats2.csv')

def evalAllResults():
    projs = get_proj_paths()
    with open("stats2.csv", "w") as f:
        f.truncate()
    for name, proj_path, _ in projs:
        evalResults("recos", proj_path, name)

def reg_and_reco(ims, in_params, config):
    name = config["name"]
    grad_width = config["grad_width"] if "grad_width" in config else (1,25)
    perf = config["perf"] if "perf" in config else False
    Ax = config["Ax"]
    method = config["method"]
    real_image = config["real_cbct"]

    print(name, grad_width)
    params = np.array(in_params[:])
    if not perf and not os.path.exists(os.path.join("recos", "forcast_"+name.split('_',1)[0]+"_reco-input.nrrd")):
        sitk.WriteImage(sitk.GetImageFromArray(real_image)*100, os.path.join("recos", "forcast_"+name.split('_',1)[0]+"_reco-input.nrrd"))
    if not perf:
        sino = sitk.GetImageFromArray(np.swapaxes(ims,0,1))
        sitk.WriteImage(sino, os.path.join("recos", "forcast_"+name+"_projs-input.nrrd"), True)
        del sino
    if not perf:# and not os.path.exists(os.path.join("recos", "forcast_"+name+"_sino-input.nrrd")):
        sino = Ax(params)
        sino = sitk.GetImageFromArray(sino)
        sitk.WriteImage(sino, os.path.join("recos", "forcast_"+name+"_sino-input.nrrd"), True)
        del sino
    if not perf:# and not os.path.exists(os.path.join("recos", "forcast_"+name+"_reco-input.nrrd")):
        reg_geo = Ax.create_geo(params)
        rec = utils.FDK_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1))
        #mask = np.zeros(rec.shape, dtype=bool)
        mask = create_circular_mask(rec.shape)
        rec = rec*mask
        del mask
        rec = sitk.GetImageFromArray(rec)*100
        sitk.WriteImage(rec, os.path.join("recos", "forcast_"+name+"_reco-input.nrrd"), True)
        del rec
    if False and not perf:# and not os.path.exists(os.path.join("recos", "forcast_"+name+"_reco-input.nrrd")):
        reg_geo = Ax.create_geo(params)
        rec = utils.CGLS_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1), 75)
        #mask = np.zeros(rec.shape, dtype=bool)
        mask = create_circular_mask(rec.shape)
        rec = rec*mask
        del mask
        rec = sitk.GetImageFromArray(rec)*100
        sitk.WriteImage(rec, os.path.join("recos", "forcast_"+name+"_reco-input_cgls.nrrd"))
        del rec
    if False and not perf:# and not os.path.exists(os.path.join("recos", "forcast_"+name+"_reco-input.nrrd")):
        reg_geo = Ax.create_geo(params)
        rec = utils.SIRT_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1), 250)
        #mask = np.zeros(rec.shape, dtype=bool)
        mask = create_circular_mask(rec.shape)
        rec = rec*mask
        del mask
        rec = sitk.GetImageFromArray(rec)*100
        sitk.WriteImage(rec, os.path.join("recos", "forcast_"+name+"_reco-input_sirt.nrrd"))
        del rec

    cali = {}
    cali['feat_thres'] = 80
    cali['iterations'] = 50
    cali['confidence_thres'] = 0.025
    cali['relax_factor'] = 0.3
    cali['match_thres'] = 60
    cali['max_ratio'] = 0.9
    cali['max_distance'] = 20
    cali['outlier_confidence'] = 85

    print_stats(config["noise"][1])
        
    perftime = time.perf_counter()
    if mp.cpu_count() > 1:
        corrs = reg_rough_parallel(ims, params, config, method)
    else:
        corrs = reg_rough(ims, params, config, method)

    vecs = Ax.create_vecs(corrs)
    write_vectors(name+"-rough-corr", corrs)
    write_vectors(name+"-rough", vecs)

    
    perftime = time.perf_counter()-perftime

    #print(params, corrs)
    if not perf:# and not os.path.exists(os.path.join("recos", "forcast_"+name+"_sino-input.nrrd")):
        sino = Ax(corrs)
        evalPerformance(sino, ims, perftime, name)
        sino = sitk.GetImageFromArray(sino)
        sitk.WriteImage(sino, os.path.join("recos", "forcast_"+name+"_sino-output.nrrd"), True)
        del sino
    
    print_stats(config["noise"][1])
    print("rough reg done ", perftime)

    if not perf:
        reg_geo = Ax.create_geo(corrs)
        #sino = Ax(corrs)
        #sino = sitk.GetImageFromArray(sino)
        #sitk.WriteImage(sino, os.path.join("recos", "forcast_"+name+"_sino-rough.nrrd"))
        #del sino
        rec = utils.FDK_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1))
        mask = create_circular_mask(rec.shape)
        rec = rec*mask
        del mask
        #rec = np.swapaxes(rec, 0, 2)
        #rec = np.swapaxes(rec, 1,2)
        #rec = rec[::-1, ::-1]
        rec = sitk.GetImageFromArray(rec)*100
        sitk.WriteImage(rec, os.path.join("recos", "forcast_"+name+"_reco-output.nrrd"), True)
        del rec
        if False:
            reg_geo = Ax.create_geo(corrs)
            rec = utils.SIRT_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1), 250)
            mask = create_circular_mask(rec.shape)
            rec = rec*mask
            del mask
            #rec = np.swapaxes(rec, 0, 2)
            #rec = np.swapaxes(rec, 1,2)
            #rec = rec[::-1, ::-1]
            rec = sitk.GetImageFromArray(rec)*100
            sitk.WriteImage(rec, os.path.join("recos", "forcast_"+name+"_reco-rough_sirt.nrrd"))
            del rec
            reg_geo = Ax.create_geo(corrs)
            rec = utils.CGLS_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1), 75)
            mask = create_circular_mask(rec.shape)
            rec = rec*mask
            del mask
            #rec = np.swapaxes(rec, 0, 2)
            #rec = np.swapaxes(rec, 1,2)
            #rec = rec[::-1, ::-1]
            rec = sitk.GetImageFromArray(rec)*100
            sitk.WriteImage(rec, os.path.join("recos", "forcast_"+name+"_reco-rough_cgls.nrrd"))
            del rec

    return vecs, corrs

def parameter_search(proj_path, cbct_path):
    ims, ims_ungained, i0s, i0s_ungained, angles, coord_systems, sids, sods = read_dicoms(proj_path, max_ims=1)

    origin, size, spacing, image = utils.read_cbct_info(cbct_path)
    real_image = utils.fromHU(sitk.GetArrayFromImage(image))

    real_image = real_image[::-1, ::-1]
    real_image = np.swapaxes(real_image, 1,2)
    real_image = np.swapaxes(real_image, 0, 2)

    detector_shape = np.array((1920,2480))
    detector_mult = np.floor(detector_shape / np.array(ims_ungained.shape[1:]))
    detector_shape = np.array(ims_ungained.shape[1:])
    detector_spacing = np.array((0.125, 0.125)) * detector_mult

    cali = {}
    cali['feat_thres'] = 80
    cali['iterations'] = 50
    cali['confidence_thres'] = 0.025
    cali['relax_factor'] = 0.3
    cali['match_thres'] = 60
    cali['max_ratio'] = 0.9
    cali['max_distance'] = 20
    cali['outlier_confidence'] = 85

    geo, (prims, secs), _ = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods, sids-sods, 1.2/np.min(spacing))

    #print(angles, prims, secs)

    input_sino = np.swapaxes(ims,0,1)
    sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join("recos", "forcast_input.nrrd"))

    Ax = utils.Ax_geo_astra(real_image.shape, real_image)
    sino = Ax(geo)
    sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino.nrrd"))

    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], geo['Vectors'][0:1])
    proj_d = forcast.Projection_Preprocessing(Ax(geo_d))[:,0]
    real_img = forcast.Projection_Preprocessing(ims[0])
    cur, _scale = forcast.roughRegistration(np.array([0,0,0,0,0.0]), real_img, proj_d, {'feat_thres': cali['feat_thres']}, geo['Vectors'][0])
    vec = forcast.applyChange(geo['Vectors'][0], cur)
    vecs = np.array([vec])
    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
    proj_d = forcast.Projection_Preprocessing(Ax(geo_d))
    sitk.WriteImage(sitk.GetImageFromArray(proj_d), os.path.join("recos", "forcast_rough.nrrd"))

    with open('stats.csv','w') as f:
        good_values= [
            [7,0.2,0.01],
            [0.5,7,0.1],
            [0.2,1,0.08],
            [5,7.33333333,0.05],
            [5,17,0.05],
        ]
        for xy,z,r in good_values:
            for i in range(len(ims)):
                print("Projection ", i)
                funs = []
                try:
                    perftime = time.perf_counter()
                    bfgs_vecs, fun, err = forcast.bfgs(i, ims, real_image, cali, geo, real_image.shape, Ax, np.array([xy, xy, z, r, r]))
                    perftime = time.perf_counter()-perftime
                    print(int(fun), int(err), xy, z, r, perftime)
                    funs.append([xy,z,r,fun,err,perftime])
                    f.write(",".join([str(e) for e in [xy,z,r,fun,err,perftime]])+"\n")
                    if fun < 12000:
                        bfgs_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([bfgs_vecs]))
                        sino = Ax(bfgs_geo)
                        #sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join("recos", "forcast_input.nrrd"))
                        sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino_bfgs--"+str(fun)+"--"+str(xy)+"_"+str(z)+"_"+str(r)+".nrrd"))
                except Exception as ex:
                    print(ex)
            for xy in itertools.chain(np.linspace(0.1,1,10), np.linspace(1,10,10)):
                #for m1 in np.linspace(3,100, 5):
                for z in itertools.chain(np.linspace(0.1,1,10), np.linspace(1,10,10)):
                    for r in itertools.chain(np.linspace(0.01,0.1,10), np.linspace(0.1,10,10)):
                        #print(e, m)
                        try:
                            perftime = time.perf_counter()
                            bfgs_vecs, fun, err = forcast.bfgs(i, ims, real_image, cali, geo, real_image.shape, Ax, np.array([xy,xy,z,r,r]))
                            perftime = time.perf_counter()-perftime
                            print(int(fun), int(err), xy, z, r, perftime)
                            funs.append([xy,z,r,fun,err,perftime])
                            f.write(",".join([str(e) for e in [xy,z,r,fun,err,perftime]])+"\n")
                            if fun < 12000:
                                bfgs_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([bfgs_vecs]))
                                sino = Ax(bfgs_geo)
                                #sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join("recos", "forcast_input.nrrd"))
                                sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino_bfgs--"+str(fun)+"--"+str(xy)+"_"+str(z)+"_"+str(r)+".nrrd"))
                        except Exception as ex:
                            print(ex)
        #with open('stats.csv','w') as f:
            #f.writelines([",".join([str(e) for e in line])+"\n" for line in funs])
        my_vecs = forcast.FORCAST(i, ims, real_image, cali, geo, real_image.shape, np.array([2.3,2.3*3,2.3,2.3*0.1,2.3*0.1]), Ax)
        my_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([my_vecs]))
        sino = Ax(my_geo)
        sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join("recos", "forcast_sino_my.nrrd"))

    rec = utils.FDK_astra(real_image.shape, geo)(np.swapaxes(ims, 0,1))
    rec = np.swapaxes(rec, 0, 2)
    rec = np.swapaxes(rec, 1,2)
    rec = rec[::-1, ::-1]
    sitk.WriteImage(sitk.GetImageFromArray(rec), os.path.join("recos", "forcast_reco.nrrd"))

def i0_est(real_img, proj_img):
    real = float(np.mean(real_img))
    proj = np.mean(proj_img)
    i0 = real * np.exp(proj)
    return i0

def get_proj_paths():
    projs = []

    if os.path.exists("E:\\output"):
        prefix = r"E:\output"
    else:
        prefix = r"D:\lumbal_spine_13.10.2020\output"
    
    cbct_path = prefix + r"\CKM4Baltimore2019\20191108-081024.994000\DCT Head Clear Nat Fill Full HU Normal [AX3D]"
    projs += [
    #('191108_balt_cbct_', prefix + '\\CKM4Baltimore2019\\20191108-081024.994000\\20sDCT Head 70kV', cbct_path),
    #('191108_balt_all_', prefix + '\\CKM4Baltimore2019\\20191108-081024.994000\\DR Overview', cbct_path),
    ]
    cbct_path = prefix + r"\CKM4Baltimore2019\20191107-091105.486000\DCT Head Clear Nat Fill Full HU Normal [AX3D]"
    projs += [
    #('191107_balt_sin1_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\Sin1', cbct_path),
    #('191107_balt_sin2_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\Sin2', cbct_path),
    #('191107_balt_sin3_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\Sin3', cbct_path),
    #('191107_balt_cbct_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\20sDCT Head 70kV', cbct_path),
    ]
    cbct_path = prefix + r"\CKM_LumbalSpine\20201020-151825.858000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    #cbct_path = prefix + r"\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    projs += [
    ('genA_trans', prefix+'\\gen_dataset\\only_trans', cbct_path),
    ('genA_angle', prefix+'\\gen_dataset\\only_angle', cbct_path),
    ('genA_both', prefix+'\\gen_dataset\\noisy', cbct_path),
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path),
    #('201020_imbu_sin_', prefix + '\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD', cbct_path),
    #('201020_imbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD', cbct_path),
    #('201020_imbu_circ_', prefix + '\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD', cbct_path),
    #('201020_imbureg_noimbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV', cbct_path),
    #('201020_imbureg_noimbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD', cbct_path),
    ]
    
    cbct_path = prefix + r"\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    projs += [
    ('genB_trans', prefix+'\\gen_dataset\\only_trans', cbct_path),
    ('genB_angle', prefix+'\\gen_dataset\\only_angle', cbct_path),
    ('genB_both', prefix+'\\gen_dataset\\noisy', cbct_path),
    #('2010201_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path),
    #('2010201_imbu_sin_', prefix + '\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD', cbct_path),
    #('2010201_imbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD', cbct_path),
    #('2010201_imbu_circ_', prefix + '\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD', cbct_path),
    #('2010201_imbureg_noimbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV', cbct_path),
    #('2010201_imbureg_noimbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD', cbct_path),
    ]
    cbct_path = prefix + r"\CKM4Baltimore\CBCT_2021_01_11_16_04_12"
    projs += [
    #('210111_balt_cbct_', prefix + '\\CKM4Baltimore\\CBCT_SINO', cbct_path),
    #('210111_balt_circ_', prefix + '\\CKM4Baltimore\\Circle_Fluoro', cbct_path),
    ]
    cbct_path = prefix + r"\CKM\CBCT\20201207-093148.064000-DCT Head Clear Nat Fill Full HU Normal [AX3D]"
    projs += [
    #('201207_cbct_', prefix + '\\CKM\\CBCT\\20201207-093148.064000-20sDCT Head 70kV', cbct_path),
    #('201207_circ_', prefix + '\\CKM\\Circ Tomo 2. Versuch\\20201207-105441.287000-P16_DR_HD', cbct_path),
    #('201207_eight_', prefix + '\\CKM\\Eight die Zweite\\20201207-143732.946000-P16_DR_HD', cbct_path),
    #('201207_opti_', prefix + '\\CKM\\Opti Traj\\20201207-163001.022000-P16_DR_HD', cbct_path),
    #('201207_sin_', prefix + '\\CKM\\Sin Traj\\20201207-131203.754000-P16_Card_HD', cbct_path),
    #('201207_tomo_', prefix + '\\CKM\\Tomo\\20201208-110616.312000-P16_DR_HD', cbct_path),
    ]
    return projs

def reg_real_data():
    projs = get_proj_paths()

    np.seterr(all='raise')

    for name, proj_path, cbct_path in projs:
        
        try:
            _, ims_un, _, _, _, coord_systems, sids, sods = read_dicoms(proj_path)
            #ims = ims[:20]
            #coord_systems = coord_systems[:20]
            skip = max(1, int(len(ims_un)/10))
            random = np.random.default_rng(23)
            #angles_noise = random.normal(loc=0, scale=0.5, size=(len(ims), 3))#*np.pi/180
            angles_noise = random.uniform(low=-1, high=1, size=(len(ims_un),3))
            angles_noise = np.zeros_like(angles_noise)
            trans_noise = random.normal(loc=0, scale=3, size=(len(ims_un), 3))

            #skip = 4
            #ims = ims[::skip]
            ims_un = ims_un[::skip]
            coord_systems = coord_systems[::skip]
            sids = np.mean(sids[::skip])
            sods = np.mean(sods[::skip])
            angles_noise = angles_noise[::skip]
            trans_noise = trans_noise[::skip]

            origin, size, spacing, image = utils.read_cbct_info(cbct_path)

            detector_shape = np.array((1920,2480))
            detector_mult = np.floor(detector_shape / np.array(ims_un.shape[1:]))
            detector_shape = np.array(ims_un.shape[1:])
            detector_spacing = np.array((0.125, 0.125)) * detector_mult

            real_image = utils.fromHU(sitk.GetArrayFromImage(image))
            del image

            Ax = utils.Ax_param_asta(real_image.shape, detector_spacing, detector_shape, sods, sids-sods, 1.2/np.min(spacing), real_image)
            Ax_gen = (real_image.shape, detector_spacing, detector_shape, sods, sids-sods, 1.2/np.min(spacing), real_image)
            geo = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods, sids-sods, 1.2/np.min(spacing))

            r = utils.rotMat(90, [1,0,0]).dot(utils.rotMat(-90, [0,0,1]))

            params = np.zeros((len(geo['Vectors']), 3, 3), dtype=float)
            params[:,1] = np.array([r.dot(v) for v in geo['Vectors'][:, 6:9]])
            params[:,2] = np.array([r.dot(v) for v in geo['Vectors'][:, 9:12]])

            #for i, (α,β,γ) in enumerate(angles_noise):
            #    params[i] = cal.applyRot(params[i], -α, -β, -γ)

            #for i, (x,y,z) in enumerate(trans_noise):
            #    params[i] = cal.applyTrans(params[i], x, y, z*5)

            projs = Ax(params)
            #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/projs.nrrd")
            #sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(ims,0,1)), "recos/ims.nrrd")

            i0s = [i0_est(ims_un[i], projs[:,i]) for i in range(ims_un.shape[0])]
            ims = -np.log(ims_un/np.mean(i0s))
            
            config = {"Ax": Ax, "Ax_gen": Ax_gen, "method": 3, "name": name, "real_cbct": real_image}

            #for method in [3,4,5,0,6]:
            for method in [0,3,4,5,7,8,9]:#,0,5]:
                config["name"] = name + str(method)
                config["method"] = method
                config["noise"] = (np.zeros((len(ims),3)), np.array(angles_noise))
                vecs, corrs = reg_and_reco(ims, np.array(params), config)
        except Exception as e:
            print(name, "cali failed", e)
            raise

if __name__ == "__main__":
    #import cProfile, io, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    reg_real_data()
    #profiler.disable()
    #s = io.StringIO()
    #sortby = pstats.SortKey.TIME
    #ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    #ps.print_stats(20)
    #print(s.getvalue())