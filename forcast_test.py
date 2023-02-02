import numpy as np
import SimpleITK as sitk
#import forcast
import cal
import utils
from utils import bcolors
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
from feature_matching import Projection_Preprocessing
from skimage.metrics import structural_similarity,normalized_root_mse
import skimage.measure
import skimage.morphology
from datetime import timedelta as td
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cal_bfgs_full
import cal_bfgs_rot
import cal_bfgs_trans
import cal_bfgs_both

def write_vectors(name, vecs):
    with open("csv\\"+name+".csv", "w") as f:
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
    #skip = 2

    edges = 30

    #sel = np.zeros(images.shape, dtype=bool)
    #sel[:,20*skip:-20*skip:skip,20*skip:-20*skip:skip] = True
    #norm_images_gained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    #norm_images_ungained = np.zeros((images.shape[0],images[0,20*skip:-20*skip:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,20*skip:-20*skip:skip].shape[0]))), dtype=np.float32)
    #sel[:,edges:-edges:skip,edges:-edges:skip] = True
    #sel_shape = (images.shape[0],images[0,edges:-edges].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,edges:-edges].shape[0])))
    #norm_images_gained = np.zeros(sel_shape, dtype=np.float32)
    #sel_shape = (images.shape[0],images[0,edges:-edges:skip].shape[0], int(np.count_nonzero(sel)/images.shape[0]/(images[0,edges:-edges:skip].shape[0])))
    #norm_images_ungained = np.zeros(sel_shape, dtype=np.float32)
    
    #gained_images = np.array([image*(1+gain/100) for image,gain in zip(images[sel].reshape(sel_shape), percent_gain)])
    
    #for i in range(len(fs)):
    #    norm_img = gained_images[i].reshape(norm_images_gained[0].shape)
    #    norm_images_gained[i] = norm_img
    #    norm_img = images[i][sel[i]].reshape(norm_images_gained[0].shape)
    #    norm_images_ungained[i] = norm_img

    oskip = 4
    if images.shape[2] > 2000:
        oskip = 4
    if edges <= 0:
        norm_images_gained = np.array([image*(1+gain/100) for image,gain in zip(images[:,::oskip,::oskip], percent_gain)])
        norm_images_ungained = np.array([image*(1+gain/100) for image,gain in zip(images[:,::skip,::skip], percent_gain)])
    else:
        norm_images_gained = np.array([image*(1+gain/100) for image,gain in zip(images[:,edges:-edges:oskip,edges:-edges:oskip], percent_gain)])
        norm_images_ungained = np.array([image*(1+gain/100) for image,gain in zip(images[:,edges:-edges:skip,edges:-edges:skip], percent_gain)])
        
    #norm_images_ungained = images[:,edges:-edges:skip, edges:-edges:skip]
    
    gain = 2.30
    #gain = 1
    offset = 400
    use = fs>1
    while (np.max(norm_images_gained, axis=(1,2))[use] > (gain*fs[use])).all():
        gain += 1
    #gain = np.ones_like(fs)
    #offset = 0
    if False:
        for i in range(len(fs)):
            norm_img = norm_images_gained[i] / (offset + (gain*fs)[i])
            #norm_img = norm_images_gained[i] / (1.1*np.max(norm_images_gained[i]))
            if (norm_img==0).any():
                norm_img[norm_img==0] = np.min(norm_img[norm_img!=0])
            norm_images_gained[i] = -np.log(norm_img)

    return norm_images_gained, norm_images_ungained, offset+gain*fs, fs

def read_dicoms(indir, max_ims=np.inf):
    print("read dicoms", indir)
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
    cs_interpol = []
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
                if len(ims) == 0:
                    ims = ds.pixel_array[2:668]
                else:
                    ims = np.vstack([ims, ds.pixel_array])

                thetas.append(float(ds.PositionerPrimaryAngle))
                phis.append(float(ds.PositionerSecondaryAngle))

                stparmdata = utils.unpack_sh_stparm(ds[0x0021,0x1012].value)

                cs = np.array(stparmdata["COORD_SYS_C_ARM"]).reshape((3, 4))
                #cs[:,2] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,2])
                #cs[:,1] = utils.rotMat(i*360/ims.shape[0], cs[:,0]).dot(cs[:,1])

                coord_systems.append(cs)
                cs_interpol.append([cs, float(ds.PositionerPrimaryAngle), float(ds.PositionerSecondaryAngle), int(ds.NumberOfFrames)])

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

                for i in range(int(ds.NumberOfFrames)):
                    ts.append(len(ts))
                    kvs.append(float(ds.KVP))
                    mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)
                    if ds[0x0021,0x1004].VR == "SL":
                        μas.append(float(ds[0x0021,0x1004].value)*0.001)
                    else:
                        μas.append(struct.unpack("<l", ds[0x0021,0x1004].value)[0]*0.001)
                    
                    sids.append(np.array(stparmdata["SID_A"]))
                    sods.append(np.array(stparmdata["SOD_A"]))
                    
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
    ims = np.array(ims)
    sids = np.array(sids)
    sods = np.array(sods)

        
        #ims = ims[:no_ims]
        #kvs = kvs[:no_ims]
        #mas = mas[:no_ims]
        #μas = μas[:no_ims]
        #percent_gain = percent_gain[:no_ims]
        #sids = sids[:no_ims]
        #sods = sods[:no_ims]

    thetas = np.array(thetas)
    phis = np.array(phis)
    
    coord_systems = np.array(coord_systems)

    cs = coord_systems
    
    if False:
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

    #if len(cs_interpol) > 0:
    #    coord_systems = np.array(cs_interpol)

    return ims_gained, ims_ungained, mas, kvs, angles, coord_systems, sids, sods

def reg_all(ims, params, config, c=0):
    noise = config["noise"]
    real_img = cal.Projection_Preprocessing(ims)
    config["real_img"] = real_img
    config["noise"] = (noise[0], noise[1])
    target_sino = config["target_sino"]
    try:
        #cur, noise = cal.bfgs_trans_all(params, config, c)
        if config["estimate"]:
            cur, _rots = cal.est_position(params[0], config["Ax"], real_img)
        else:
            if c <= -60:
                cur, noise = cal_bfgs_trans.bfgs(params, config, c)
            elif c <= -40:
                config["real_img"] = real_img[:len(params)//2]
                config["noise"] = (noise[0][:len(params)//2], noise[1][:len(params)//2])
                config["target_sino"] = target_sino[:,:len(params)//2]
                cur1, noise1 = cal_bfgs_full.bfgs_single(params[:len(params)//2], config, c)
                config["real_img"] = real_img[len(params)//2:]
                config["noise"] = (noise[0][len(params)//2:], noise[1][len(params)//2:])
                config["target_sino"] = target_sino[:,len(params)//2:]
                cur2, noise2 = cal_bfgs_full.bfgs_single(params[len(params)//2:], config, c)
                cur = np.concatenate((cur1,cur2))
                noise = np.concatenate((noise1,noise2))
            else:
                config["real_img"] = real_img[:len(params)//2]
                config["noise"] = (noise[0][:len(params)//2], noise[1][:len(params)//2])
                config["target_sino"] = target_sino[:,:len(params)//2]
                cur1, noise1 = cal_bfgs_rot.bfgs_single(params[:len(params)//2], config, c)
                config["real_img"] = real_img[len(params)//2:]
                config["noise"] = (noise[0][len(params)//2:], noise[1][len(params)//2:])
                config["target_sino"] = target_sino[:,len(params)//2:]
                cur2, noise2 = cal_bfgs_rot.bfgs_single(params[len(params)//2:], config, c)
                cur = np.concatenate((cur1,cur2))
                noise = np.concatenate((noise1,noise2))
    except Exception as ex:
        print(ex)
        raise
    
    corrs = np.array(cur)
    config["noise"] = noise
    return corrs

def reg_rough(ims, params, config, c=0):
    corrs = [None]*len(params)
    noise = config["noise"]
    config["log_queue"] = mp.Queue()
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
                if config["estimate"]:
                    if "est_data" not in config or config["est_data"] is None:
                        config["est_data"] = cal.simulate_est_data(cur, config["Ax"])
                    cur, _rots = cal.est_position(cur, config["Ax"], [real_img], config["est_data"])
                    cur = cur[0]
                else:
                    if c >= 0:
                        cur = cal.roughRegistration(cur, config, c)
                    else:
                        cur = cal_bfgs_both.bfgs(cur, config, c)
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

def it_func(con, Ax_params, log_queue, ready, name):
    if name != None:
        profiler = cProfile.Profile()
    try:
        
        #print("start")
        np.seterr(all='raise')
        Ax = None
        Ax = utils.Ax_param_asta(*Ax_params)
        est_data = None
        while True:
            try:
                con.send(("ready",))
                ready.set()
                (i, cur, im, target_sino, noise, method) = con.recv()
                #print(i)
                old_stdout = sys.stdout
                sys.stdout = stringout = io.StringIO()

                if name != None:
                    profiler.enable()    
                real_img = cal.Projection_Preprocessing(im)
                if method == "estimate":
                    if est_data is None:
                        est_data = cal.simulate_est_data(cur, Ax)
                cur_config = {"real_img": real_img, "Ax": Ax, "log_queue": log_queue, "name": str(i), "target_sino": target_sino, "est_data": est_data}
                try:
                    if method >= 0:
                        cur = cal.roughRegistration(cur, cur_config, method)
                    else:
                        cur = cal_bfgs_both.bfgs(cur, cur_config, method)
                except Exception as ex:
                    print(ex, type(ex), i, cur, file=sys.stderr)
                    con.send(("error",))
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
        exit()

def it_log(log_queue):
    while True:
        try:
            name, value = log_queue.get()
            if name == "exit":
                return
            with open("csv\\"+name+".csv", "a") as f:
                f.write("{};".format(value))
        except Exception as ex:
            print("logger faulty: ", ex)

def reg_rough_parallel(ims, params, config, c=0):
    corrs = []
    pool_size = mp.cpu_count()
    #if c==28:
    #    pool_size = 2
    #elif c >= 40:
    #    pool_size = mp.cpu_count()-4
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

    log_queue = mp.Queue()
    log_proc = mp.Process(target=it_log, args=(log_queue,), daemon=True)
    log_proc.start()

    while np.array([e is None for e in corrs]).any(): #len(indices)>0:
        ready_con = None
        while ready_con is None:
            for _ in range(len(pool), pool_size):
                p = mp.Pipe(True)
                if "profile" in config and config["profile"]:
                    name = config["name"]+"_"+str(proc_count)
                else:
                    name = None
                proc = mp.Process(target=it_func, args=(p[1], config["Ax_gen"], log_queue, ready, name), daemon=True)
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
                                exit(0)
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
            if config["estimate"]:
                ready_con[0].send((i, params[i], ims[i], config["target_sino"][:,i], (config["noise"][0][i],config["noise"][1][i]), "estimate"))
            else:
                ready_con[0].send((i, params[i], ims[i], config["target_sino"][:,i], (config["noise"][0][i],config["noise"][1][i]), c))
            ready_con[3] = i

    for con in pool:
        con[2].terminate()
        con[0].close()
        con[1].close()

    log_queue.put(("exit", 0))
        
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

margin=50

def evalNeedleArea(img, img2, projname, name):
    if False and projname in ("genA", "201020"):
        if img.shape[0] > 500:
            pos = [408, 472-margin, 498-margin]
            mult = 2
        else:
            pos = [204, 236-margin, 249-margin]
            mult = 1
    elif True or projname in ("genB", "2010201"):
        if img.shape[0] > 500:
            pos = [465, 514-margin, 525-margin]
            mult = 2
        else:
            pos = [232, 257-margin, 262-margin]
            pos = [199, 258-margin, 257-margin]
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
    mask = skimage.morphology.binary_opening(mask*(img>(maxval*0.5)), skimage.morphology.cube(3))
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
    mask = skimage.morphology.binary_opening(mask*(img2>(maxval*0.5)), skimage.morphology.cube(3))
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

def evalPerformance(output, real, runtime, name, stats_file='stats.csv', real_fwhm=None, real_area=None, gi_config={"GIoldold":[None], "absp1":[None], "p1":[None]}):
    if np.size(output[0]) != np.size(real[0]):
        return
    vals = []
    #for i in range(len(real)):
    if "rec" in stats_file:
        vals.append(1.0/cal.calcGIObjective(real, output, 0, None, gi_config))
    else:
        for i in range(real.shape[0]):
            vals.append(1.0/cal.calcGIObjective(real[i], output[i], 0, None, {"GIoldold":[None], "absp1":[None], "p1":[None]}))
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
            vals7.append(normalized_root_mse(real[i], output[i]))
    print("NRMSE: ", np.mean(vals7), np.median(vals7))

    vals8 = []
    #if "rec" in stats_file:
        #vals8.append(evalFWHM(output, "genB"))
        #if real_fwhm is None:
        #    real_fwhm = evalFWHM(real, "genB")
        #vals8.append(vals8[-1]-real_fwhm)
    
    vals9 = []
    if "rec" in stats_file:
        vals9.append(evalNeedleArea(output, real, "genB", name))
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
            f.write(";".join([str(d) for d in [name, runtime/(24*60*60), np.mean(vals), np.mean(vals6), np.mean(vals7), np.mean(vals9)]]) + "\n")
        else:
            f.write(";".join([str(d) for d in [name, runtime/(24*60*60), np.mean(vals), np.mean(vals6), np.mean(vals7)]]) + "\n")

def evalSinoResults(out_path, in_path, projname):
    ims, ims_un, mas, kvs, angles, coord_systems, sids, sods = read_dicoms(in_path)
    detector_shape = np.array((1920,2480))
    detector_mult = int(np.floor(detector_shape[0] / ims_un.shape[1]))
    detector_edges = int(detector_shape[0] / detector_mult - ims_un.shape[1]), int(detector_shape[1] / detector_mult - ims_un.shape[2])
    i0_ims, i0_mas, i0_kvs = i0_data(detector_mult, (detector_mult//2)*detector_edges[0])
    i0s = i0_interpol(i0_ims, i0_mas, np.mean(mas))
    ims_norm = np.array(-np.log(ims_un/i0s), dtype=np.float32)

    for filename in os.listdir(out_path):
        if re.fullmatch("target_sino.nrrd", filename) != None:
            img = sitk.ReadImage(os.path.join(out_path, filename))
            proj = sitk.GetArrayFromImage(img)
            if proj.dtype != np.float32:
                proj = np.array(proj, dtype=np.float32)
            ims_norm2 = np.swapaxes(proj, 0, 1)


    def sino_data():
        input_sino = False
        for filename in os.listdir(out_path):
            if re.fullmatch("forcast_(.+?)_sino-output.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                if proj.dtype != np.float32:
                    proj = np.array(proj, dtype=np.float32)
                #projs.append(proj)
                #names.append("_".join(filename.split('_')[1:-1]))
                yield "_".join(filename.split('_')[1:-1]), proj
            if re.fullmatch("forcast_(.+?)_sino-input.nrrd", filename) != None and projname in filename and input_sino == False:
                input_sino = True
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                if proj.dtype != np.float32:
                    proj = np.array(proj, dtype=np.float32)
                #projs.append(proj)
                #names.append(projname+"_input")
                yield projname+"_input", proj
            if re.fullmatch("trajtomo_reco_matlab_(.+?)_output.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                if proj.dtype != np.float32:
                    proj = np.array(proj, dtype=np.float32)
                #projs.append(proj)
                #names.append("_".join(filename.split('_')[3:-1])+"_matlab")
                yield "_".join(filename.split('_')[3:-1])+"_matlab", proj

    for name, proj in sino_data():
        print(name)
        skip = int(np.ceil(ims_un.shape[0]/proj.shape[1]))

        #i0s = [i0_est(ims_un[::skip][i], proj[:,i]) for i in range(proj.shape[1])]
        #ims = np.array(-np.log(ims_un[::skip]/np.mean(i0s)), dtype=np.float32)

        #i0s = i0_interpol(i0_ims, i0_mas, np.mean(mas))
        #ims = -np.log(ims/i0s)[::skip]
        ims = ims_norm[::skip]
        ims2 = ims_norm2[::skip]
        try:
            evalPerformance(np.swapaxes(proj, 0, 1), ims, 0, name, 'stats_proj.csv')
            evalPerformance(np.swapaxes(proj, 0, 1), ims2, 0, name+"sim", 'stats_proj.csv')
        except Exception as e:
            print(e)
    
def evalRecoResults(out_path, in_path, projname):
    global out_rec_meta
    input_recos = {}
    input_gi_confs = {}
    input_fwhm = {}
    input_area = {}
    metas = {}
    for filename in os.listdir(out_path):
        #if re.fullmatch("forcast_[^_]+_reco-input.nrrd", filename) != None:
        if re.fullmatch("forcast_201020(1_imbu|_imbureg_noimbu)_cbct_4_reco-output.nrrd", filename) != None:
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
            if re.fullmatch("forcast_matlab_(.+?)_reco-output.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[2:]))
                yield "_".join(filename.split('_')[2:]), proj, filename
            elif re.fullmatch("forcast_matlab_(.+?)_reco-output_sirt.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[2:]))
                yield "_".join(filename.split('_')[2:]), proj, filename
            elif re.fullmatch("forcast_matlab_(.+?)_reco-output_cgls.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[2:]))
                yield "_".join(filename.split('_')[2:]), proj, filename
            elif re.fullmatch("forcast_(.+?)_reco-output_sirt.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[1:]))
                yield "_".join(filename.split('_')[1:]), proj, filename
            elif re.fullmatch("forcast_(.+?)_reco-output_cgls.nrrd", filename) != None and projname in filename:
                img = sitk.ReadImage(os.path.join(out_path, filename))
                proj = sitk.GetArrayFromImage(img)
                #projs.append(proj)
                #print(filename, proj.shape)
                #names.append("_".join(filename.split('_')[1:]))
                yield "_".join(filename.split('_')[1:]), proj, filename
            elif re.fullmatch("forcast_(.+?)_reco-output.nrrd", filename) != None and projname in filename:
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
        real_name = "2010201"
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
                mask = create_circular_mask(proj.shape, radius_off=margin)
                #write_images(1.0*mask, "mask.nrrd")
                output = (proj*mask)[:,margin:-margin,margin:-margin]
                real = (ims*mask)[:,margin:-margin,margin:-margin]
                print(output.shape, real.shape)
                input_gi_confs[real_name] = {"GIoldold":[None], "absp1":[None], "p1":[None]}
                evalPerformance(output, real, 0, name, 'stats_rec.csv', real_fwhm=input_fwhm[real_name], real_area=input_area[real_name], gi_config=input_gi_confs[real_name])

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
    projs = get_proj_paths()
    if evalSino:
        with open("stats_proj.csv", "w") as f:
            f.truncate()
            f.write(";".join(["Name", "Runtime", "NGI", "SSIM", "NRMSE"]) + "\n")
        for name, proj_path, _, _ in projs:
            print(proj_path)
            evalSinoResults(outpath, proj_path, name)
    if evalReco:        
        with open("stats_rec.csv", "w") as f:
            f.truncate()
            f.write(";".join(["Name", "Runtime", "NGI", "SSIM", "NRMSE", "Area"]) + "\n")
        for name, proj_path, _, _ in projs:
            print(proj_path)
            evalRecoResults(outpath, proj_path, name)

def write_rec(geo, ims, filepath, mult=1):
    #geo["Vectors"] = geo["Vectors"]*mult
    mult = int(np.round(ims.shape[1] / geo['DetectorRowCount']))
    geo = astra.create_proj_geom('cone_vec', ims.shape[1], ims.shape[2], geo["Vectors"]*mult)
    out_shape = (out_rec_meta[3][0]*mult, out_rec_meta[3][1]*mult, out_rec_meta[3][2]*mult)
    rec = utils.FDK_astra(out_shape, geo, np.swapaxes(ims, 0,1))
    #mask = np.zeros(rec.shape, dtype=bool)
    mask = create_circular_mask(rec.shape)
    rec = rec*mask
    del mask
    write_images(rec, filepath, mult)
    return

    rec = utils.SIRT_astra(out_shape, geo, np.swapaxes(ims, 0,1), 200)
    #mask = np.zeros(rec.shape, dtype=bool)
    mask = create_circular_mask(rec.shape)
    rec = rec*mask
    del mask
    write_images(rec, filepath.rsplit('.', maxsplit=1)[0]+"_sirt."+filepath.rsplit('.', maxsplit=1)[1], mult)

    rec = utils.CGLS_astra(out_shape, geo, np.swapaxes(ims, 0,1), 50)
    #mask = np.zeros(rec.shape, dtype=bool)
    mask = create_circular_mask(rec.shape)
    rec = rec*mask
    del mask
    write_images(rec, filepath.rsplit('.', maxsplit=1)[0]+"_cgls."+filepath.rsplit('.', maxsplit=1)[1], mult)

def write_images(rec, filepath, mult=1):
    rec = sitk.GetImageFromArray(rec)#*100
    rec.SetOrigin(out_rec_meta[0])
    out_spacing = (out_rec_meta[2][0]/mult,out_rec_meta[2][1]/mult,out_rec_meta[2][2]/mult)
    rec.SetSpacing(out_spacing)
    sitk.WriteImage(rec, filepath, True)
    del rec
    
def reg_and_reco(ims_big, ims, in_params, config):
    name = config["name"]
    grad_width = config["grad_width"] if "grad_width" in config else (1,25)
    perf = config["perf"] if "perf" in config else False
    Ax = config["Ax"]
    method = config["method"]
    use_saved = config["use_saved"] if "use_saved" in config else False
    real_image = config["real_cbct"]
    outpath = config["outpath"]

    print(name, grad_width)
    params = np.array(in_params[:])
    if not perf and not os.path.exists(os.path.join(outpath, "forcast_"+name.split('_',1)[0]+"_reco-input.nrrd")):
        rec = sitk.GetImageFromArray(real_image)*100
        rec.SetOrigin(out_rec_meta[0])
        out_spacing = (out_rec_meta[2][0],out_rec_meta[2][1],out_rec_meta[2][2])
        rec.SetSpacing(out_spacing)
        sitk.WriteImage(rec, os.path.join(outpath, "forcast_"+name.split('_',1)[0]+"_reco-input.nrrd"))
    if not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_projs-input.nrrd")):
        sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(ims,0,1)))
        sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_projs-input.nrrd"), True)
        del sino
    if False and not perf and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_reco-input.nrrd")):
        reg_geo = Ax.create_geo(params)
        write_rec(reg_geo, ims_big, os.path.join(outpath, "forcast_"+name+"_reco-input.nrrd"))
    if not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_sino-input.nrrd")):
        sino = cal.Projection_Preprocessing(Ax(params))
        img = cv2.drawMatchesKnn(np.array(255*(ims[-1]-np.min(ims[-1]))/(np.max(ims[-1])-np.min(ims[-1])),dtype=np.uint8), None,
            np.array(255*(sino[:,-1]-np.min(sino[:,-1]))/(np.max(sino[:,-1])-np.min(sino[:,-1])),dtype=np.uint8),None, None, None)
        cv2.imwrite("img\\check_" + name + "_pre.png", img)
        sino = sitk.GetImageFromArray(sino)
        sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_sino-input.nrrd"), True)
        del sino

    cali = {}
    cali['feat_thres'] = 80
    cali['iterations'] = 50
    cali['confidence_thres'] = 0.025
    cali['relax_factor'] = 0.3
    cali['match_thres'] = 60
    cali['max_ratio'] = 0.9
    cali['max_distance'] = 20
    cali['outlier_confidence'] = 85

    if "noise" in config:
        print_stats(config["noise"][1])
        
    perftime = time.perf_counter()
    if use_saved:
        vecs = read_vectors(name+"-rough")
        corrs = read_vectors(name+"-rough-corr")
    else:
        if method>-20:# and not config["estimate"]:
            if config["paralell"] and  mp.cpu_count() > 1:
                corrs = reg_rough_parallel(ims, params, config, method)
            else:
                corrs = reg_rough(ims, params, config, method)
        else:
            corrs = reg_all(ims, params, config, method)

    vecs = Ax.create_vecs(corrs)
    write_vectors(name+"-rough-corr", corrs)
    write_vectors(name+"-rough", vecs)

    
    perftime = time.perf_counter()-perftime

    #print(params, corrs)
    if not perf:# and not os.path.exists(os.path.join(outpath, "forcast_"+name+"_sino-input.nrrd")):
        sino = Ax(corrs)
        img = cv2.drawMatchesKnn(np.array(255*(ims[-1]-np.min(ims[-1]))/(np.max(ims[-1])-np.min(ims[-1])),dtype=np.uint8), None,
            np.array(255*(sino[:,-1]-np.min(sino[:,-1]))/(np.max(sino[:,-1])-np.min(sino[:,-1])),dtype=np.uint8),None, None, None)
        cv2.imwrite("img\\check_" + name + "_post.png", img)
        sino = sitk.GetImageFromArray(sino)
        sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_sino-output.nrrd"), True)
        #evalPerformance(np.swapaxes(sino, 0, 1), ims, perftime, name)
        del sino
    
    if "noise" in config:
        print_stats(config["noise"][1])
    print("rough reg done ", perftime)

    if not perf:
        reg_geo = Ax.create_geo(corrs)
        mult = 1
        write_rec(reg_geo, ims, os.path.join(outpath, "forcast_"+name+"_reco-output.nrrd"), mult)

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
    sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join(outpath, "forcast_input.nrrd"))

    Ax = utils.Ax_geo_astra(real_image.shape, real_image)
    sino = Ax(geo)
    sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join(outpath, "forcast_sino.nrrd"))

    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], geo['Vectors'][0:1])
    proj_d = forcast.Projection_Preprocessing(Ax(geo_d))[:,0]
    real_img = forcast.Projection_Preprocessing(ims[0])
    cur, _scale = forcast.roughRegistration(np.array([0,0,0,0,0.0]), real_img, proj_d, {'feat_thres': cali['feat_thres']}, geo['Vectors'][0])
    vec = forcast.applyChange(geo['Vectors'][0], cur)
    vecs = np.array([vec])
    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
    proj_d = forcast.Projection_Preprocessing(Ax(geo_d))
    sitk.WriteImage(sitk.GetImageFromArray(proj_d), os.path.join(outpath, "forcast_rough.nrrd"))

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
                        #sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join(outpath, "forcast_input.nrrd"))
                        sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join(outpath, "forcast_sino_bfgs--"+str(fun)+"--"+str(xy)+"_"+str(z)+"_"+str(r)+".nrrd"))
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
                                #sitk.WriteImage(sitk.GetImageFromArray(input_sino), os.path.join(outpath, "forcast_input.nrrd"))
                                sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join(outpath, "forcast_sino_bfgs--"+str(fun)+"--"+str(xy)+"_"+str(z)+"_"+str(r)+".nrrd"))
                        except Exception as ex:
                            print(ex)
        #with open('stats.csv','w') as f:
            #f.writelines([",".join([str(e) for e in line])+"\n" for line in funs])
        my_vecs = forcast.FORCAST(i, ims, real_image, cali, geo, real_image.shape, np.array([2.3,2.3*3,2.3,2.3*0.1,2.3*0.1]), Ax)
        my_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([my_vecs]))
        sino = Ax(my_geo)
        sitk.WriteImage(sitk.GetImageFromArray(sino), os.path.join(outpath, "forcast_sino_my.nrrd"))

    rec = utils.FDK_astra(real_image.shape, geo)(np.swapaxes(ims, 0,1))
    rec = np.swapaxes(rec, 0, 2)
    rec = np.swapaxes(rec, 1,2)
    rec = rec[::-1, ::-1]
    sitk.WriteImage(sitk.GetImageFromArray(rec), os.path.join(outpath, "forcast_reco.nrrd"))

def i0_est(real_img, proj_img):
    filt = np.zeros_like(real_img, dtype=bool)
    filt[filt.shape[0]//3:-filt.shape[0]//3,filt.shape[1]//3:-filt.shape[1]//3] = True
    real = float(np.mean(real_img[filt]))
    p = scipy.ndimage.zoom(proj_img, np.array(real_img.shape)/np.array(proj_img.shape))
    proj = np.mean(p[filt])
    i0 = real * np.exp(proj)
    
    #i0 = i0*1.4
    i0 += 400
    return i0

def i0_data(skip, edges):
    if os.path.exists("F:\\output"):
        prefix = r"F:\output"
    elif os.path.exists(r"D:\lumbal_spine_13.10.2020\output"):
        prefix = r"D:\lumbal_spine_13.10.2020\output"
    else:
        prefix = r".\output"

    path = os.path.join(prefix, '70kVp')

    kvs = []
    mas = []
    ims = []

    for root, _dirs, files in os.walk(path):
        for entry in files:
            path = os.path.abspath(os.path.join(root, entry))
            ds = pydicom.dcmread(path)
            if edges <= 0:
                ims.append(np.mean(ds.pixel_array[:,::skip, ::skip], axis=0))
            else:
                ims.append(np.mean(ds.pixel_array[:,edges:-edges:skip, edges:-edges:skip], axis=0))
            kvs.append(float(ds.KVP))
            mas.append(float(ds.XRayTubeCurrent)*float(ds.ExposureTime)*0.001)

    kvs = np.array(kvs)
    mas = np.array(mas)
    ims = np.array(ims)

    print(ims.shape)
    return ims, mas, kvs

def i0_interpol(ims, mas, target):
    res = np.zeros_like(ims[0])

    for i in range(ims.shape[1]):
        for j in range(ims.shape[2]):
            p = np.polyfit(np.array(mas), ims[:,i,j], 1)
            res[i,j] = np.polyval(p, target)

    return res

def get_proj_paths():
    projs = []

    if os.path.exists("F:\\output"):
        prefix = r"F:\output"
    elif os.path.exists("D:\\lumbal_spine_13.10.2020\\output"):
        prefix = r"D:\lumbal_spine_13.10.2020\output"
    else:
        prefix = r".\output"
    
    cbct_path = prefix + r"\CKM4Baltimore2019\20191108-081024.994000\DCT Head Clear Nat Fill Full HU Normal [AX3D]"
    projs += [
    #('191108_balt_cbct_', prefix + '\\CKM4Baltimore2019\\20191108-081024.994000\\20sDCT Head 70kV', cbct_path, [4, 28, 29]),
    #('191108_balt_all_', prefix + '\\CKM4Baltimore2019\\20191108-081024.994000\\DR Overview', cbct_path, [4, 28, 29]),
    ]
    cbct_path = prefix + r"\CKM4Baltimore2019\20191107-091105.486000\DCT Head Clear Nat Fill Full HU Normal [AX3D]"
    projs += [
    #('191107_balt_cbct_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\20sDCT Head 70kV', cbct_path, [4, 28, 29]),
    #('191107_balt_sin1_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\Sin1', cbct_path, [4, 28, 29]),
    #('191107_balt_sin2_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\Sin2', cbct_path, [4, 28, 29]),
    #('191107_balt_sin3_', prefix + '\\CKM4Baltimore2019\\20191107-091105.486000\\Sin3', cbct_path, [4, 28, 29]),
    ]
    cbct_path = prefix + r"\CKM_LumbalSpine\20201020-151825.858000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    #cbct_path = prefix + r"\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    projs += [
    #('genA_trans', prefix+'\\gen_dataset\\only_trans', cbct_path, [4]),
    #('genA_angle', prefix+'\\gen_dataset\\only_angle', cbct_path, [4,20,21,22,23,24,25,26]),
    #('genA_both', prefix+'\\gen_dataset\\noisy', cbct_path, [4,20,21,22,23,24,25,26]),
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [50,52]), # normal noise
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-44,-58,-34,-24,33,42]), # normal noise
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-43,-57,34,41]), # reduced noise
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-73,-67]), # normal noise trans
    #('201020_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [-72,-66]), # reduced noise trans
    #('201020_imbu_sin_', prefix + '\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    #('201020_imbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    #('201020_imbu_circ_', prefix + '\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD', cbct_path, [4, -34, -35, 28, 29]),
    #('201020_imbureg_noimbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV', cbct_path, [4, 28, 29]),
    #('201020_imbureg_noimbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    ]
    
    cbct_path = prefix + r"\CKM_LumbalSpine\20201020-093446.875000\DCT Head Clear Nat Fill Full HU Normal [AX3D] 70kV"
    projs += [
    #('genB_trans', prefix+'\\gen_dataset\\only_trans', cbct_path, [4]),
    #('genB_angle', prefix+'\\gen_dataset\\only_angle', cbct_path),
    #('genB_both', prefix+'\\gen_dataset\\noisy', cbct_path),
    ('2010201_imbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\20sDCT Head 70kV', cbct_path, [33, 60, 62, 63]),
    #('2010201_imbu_sin_', prefix + '\\CKM_LumbalSpine\\20201020-122515.399000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    #('2010201_imbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-093446.875000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    #('2010201_imbu_circ_', prefix + '\\CKM_LumbalSpine\\20201020-140352.179000\\P16_DR_LD', cbct_path, [4, -34, -35, 28, 29]),
    ('2010201_imbu_arc_', prefix + '\\CKM_LumbalSpine\\Arc\\20201020-150938.350000-P16_DR_LD', cbct_path, [60,61,62,63,64,65]),
    #('2010201_noimbu_arc_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\P16_DR_LD', cbct_path, [60,61,62,63,64,65]),
    #('2010201_imbureg_noimbu_cbct_', prefix + '\\CKM_LumbalSpine\\20201020-151825.858000\\20sDCT Head 70kV', cbct_path, [4, 28, 29]),
    #('2010201_imbureg_noimbu_opti_', prefix + '\\CKM_LumbalSpine\\20201020-152349.323000\\P16_DR_LD', cbct_path, [4, 28, 29]),
    ]
    return projs
    cbct_path = prefix + r"\CKM4Baltimore\CBCT_2021_01_11_16_04_12"
    projs += [
    #('210111_balt_cbct_', prefix + '\\CKM4Baltimore\\CBCT_SINO', cbct_path, [4, 28, 29]),
    #('210111_balt_circ_', prefix + '\\CKM4Baltimore\\Circle_Fluoro', cbct_path, [4, 28, 29]),
    ]
    cbct_path = prefix + r"\CKM\CBCT\20201207-093148.064000-DCT Head Clear Nat Fill Full HU Normal [AX3D]"
    projs += [
    ('201207_arc_', prefix + '\\CKM\\ArcTest', cbct_path, [60,61,62,63,64,65])
    #('201207_cbct_', prefix + '\\CKM\\CBCT\\20201207-093148.064000-20sDCT Head 70kV', cbct_path, [4, 28, 29]),
    #('201207_circ_', prefix + '\\CKM\\Circ Tomo 2. Versuch\\20201207-105441.287000-P16_DR_HD', cbct_path, [4, 28, 29]),
    #('201207_eight_', prefix + '\\CKM\\Eight die Zweite\\20201207-143732.946000-P16_DR_HD', cbct_path, [4, 28, 29]),
    #('201207_opti_', prefix + '\\CKM\\Opti Traj\\20201207-163001.022000-P16_DR_HD', cbct_path, [4, 28, 29]),
    #('201207_sin_', prefix + '\\CKM\\Sin Traj\\20201207-131203.754000-P16_Card_HD', cbct_path, [24]),
    #('201207_tomo_', prefix + '\\CKM\\Tomo\\20201208-110616.312000-P16_DR_HD', cbct_path, [4, 28, 29]),
    ]
    cbct_path = r"D:\rothfuss\ProejctionData\test\DCT_BODY_NAT_FILL_FULL_HU_NORMAL_[AX3D]_0061"
    projs += [
        #('rothfuss_', r"D:\rothfuss\ProejctionData\test\DR_OVERVIEW_0065", cbct_path, [6]),
    ]
    cbct_path = r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DCT_BODY_NAT_FILL_FULL_HU_NORMAL_[AX3D]_0001"
    projs += [
        #('r_pre_05',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DR_OVERVIEW_0005", cbct_path, [6]),
        #('r_pre_12',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DR_OVERVIEW_0012", cbct_path, [6]),
        #('r_pre_13',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DR_OVERVIEW_0013", cbct_path, [6]),
    ]
    cbct_path = r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DCT_BODY_NAT_FILL_FULL_HU_NORMAL_[AX3D]_0017"
    projs += [
        #('r_post_05',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DR_OVERVIEW_0005", cbct_path, [6]),
        #('r_post_12',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DR_OVERVIEW_0012", cbct_path, [6]),
        #('r_post_13',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_152920_434000\DR_OVERVIEW_0013", cbct_path, [6]),
    ]
    cbct_path = r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DCT_BODY_NAT_FILL_FULL_HU_NORMAL_[AX3D]_0002"
    s = [4]
    projs += [
        ('r_pre_04_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0004", cbct_path, s+[]),
        ('r_pre_05_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0005", cbct_path, s+[]),
        ('r_pre_06_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0006", cbct_path, s+[]),
        ('r_pre_07_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0007", cbct_path, s+[]),
        ('r_pre_08_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0008", cbct_path, s+[]),
        ('r_pre_09_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0009", cbct_path, s+[]),
        ('r_pre_10_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0010", cbct_path, s+[]),
        ('r_pre_11_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0011", cbct_path, s+[]),
        ('r_pre_12_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0012", cbct_path, s+[]),
    ]
    cbct_path = r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\3D_BODY_NAT_FILL_FULL_HU_AUTO_[AX3D]_0003"
    projs += [
        ('r_mid_04_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0004", cbct_path, s+[]),
        ('r_mid_05_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0005", cbct_path, s+[]),
        ('r_mid_06_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0006", cbct_path, s+[]),
        ('r_mid_07_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0007", cbct_path, s+[]),
        ('r_mid_08_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0008", cbct_path, s+[]),
        ('r_mid_09_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0009", cbct_path, s+[]),
        ('r_mid_10_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0010", cbct_path, s+[]),
        ('r_mid_11_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0011", cbct_path, s+[]),
        ('r_mid_12_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0012", cbct_path, s+[]),
    ]
    cbct_path = r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DCT_BODY_NAT_FILL_FULL_HU_NORMAL_[AX3D]_0013"
    projs += [
        ('r_post_04_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0004", cbct_path, s+[]),
        ('r_post_05_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0005", cbct_path, s+[]),
        ('r_post_06_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0006", cbct_path, s+[]),
        ('r_post_07_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0007", cbct_path, s+[]),
        ('r_post_08_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0008", cbct_path, s+[]),
        ('r_post_09_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0009", cbct_path, s+[]),
        ('r_post_10_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0010", cbct_path, s+[]),
        ('r_post_11_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0011", cbct_path, s+[]),
        ('r_post_12_',r"D:\rothfuss\GUIDOO_MK2_TESTDATASET_GUIDOO_MK2-TESTDATASET\__20210624_154436_571000\DR_OVERVIEW_0012", cbct_path, s+[]),
    ]
    return projs

def calc_images_matlab(name, ims, real_image, detector_shape, outpath, geo):
    vecs = utils.read_matlab_vecs(name)
    from scipy.io import loadmat
    runtime = loadmat("runtime")['runtime'][0,0]

    #skip = ims_all.shape[0] // len(vecs)
    #ims = ims_all[::skip]
    
    # [0,1,2], [0,2,1], [2,0,1], [2,1,0], [1,0,2], [1,2,0]
    print(real_image.shape)
    real_image = np.transpose(real_image, axes=[0,2,1])[::-1,::,::]
    #real_image = np.transpose(real_image, axes=[0,1,2])[::,::,::]
    print(real_image.shape)
    Ax = utils.Ax_vecs_astra(real_image.shape, (detector_shape[0],detector_shape[1]), real_image)
    
    #sino = np.swapaxes(Ax(geo['Vectors']), 0,1)
    #print(sino.shape)

    #sino = np.transpose(Ax(vecs), axes=[2,1,0])[::,::,::-1]
    r = [utils.rotMat(-90, v) for v in np.cross(vecs[:, 6:9], vecs[:, 9:12])]
    vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
    vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
    vecs[:, 6:9] = -np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
    vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])
    sino = Ax(vecs)
    print(sino.shape)
    #evalPerformance(np.swapaxes(sino, 0, 1), ims, runtime, name)
    sino = sitk.GetImageFromArray(sino)
    sitk.WriteImage(sino, os.path.join(outpath, "forcast_matlab_"+name+"_sino-output.nrrd"))
    del sino
    
    if True:
        sino = sitk.GetImageFromArray(np.transpose(ims, axes=[1,0,2])[::,::,::-1])
        #sino = sitk.GetImageFromArray(np.transpose(ims, axes=[1,0,2])[::,::,::])
        sitk.WriteImage(sino, os.path.join(outpath, "forcast_matlab_"+name+"_sino-input.nrrd"))
        del sino

    if True:
        #a = vecs[:,6:9]
        #vecs[:,6:9] = -vecs[:,9:12]
        #vecs[:,9:12] = -a
        #r = [utils.rotMat(-90, v) for v in np.cross(vecs[:, 6:9], vecs[:, 9:12])]
        #vecs[:, 0:3] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 0:3])])
        #vecs[:, 3:6] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 3:6])])
        #vecs[:, 6:9] = -np.array([r.dot(v) for r,v in zip(r, vecs[:, 6:9])])
        #vecs[:, 9:12] = np.array([r.dot(v) for r,v in zip(r, vecs[:, 9:12])])
        reg_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        #write_rec(reg_geo, np.transpose(ims, axes=[2,0,1])[::-1,::,::], os.path.join(outpath, "forcast_matlab_"+name+"_reco-output.nrrd"), mult=1)
        filepath = os.path.join(outpath, "forcast_matlab_"+name+"_reco-output.nrrd")
        mult = 1
        reg_geo["Vectors"] = reg_geo["Vectors"]*mult
        out_shape = (out_rec_meta[3][0]*mult, out_rec_meta[3][1]*mult, out_rec_meta[3][2]*mult)
        #sino = Ax(vecs)
        #proj_id = astra.data3d.create('-proj3d', reg_geo, np.transpose(ims, axes=[2,0,1])[::-1,::,::])
        proj_id = astra.data3d.create('-proj3d', reg_geo, np.transpose(ims, axes=[1,0,2])[::,::,::-1])
        vol_geom = astra.create_vol_geom(out_shape[1], out_shape[2], out_shape[0])
        rec_id = astra.data3d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['Option'] = {#"VoxelSuperSampling": 3, 
                        #"ShortScan": True
                        }
        alg_id = astra.algorithm.create(cfg)
        iterations = 150
        astra.algorithm.run(alg_id, iterations)
        rec = astra.data3d.get(rec_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(rec_id)
        astra.algorithm.delete(alg_id)
        #rec = utils.FDK_astra(out_shape, reg_geo, np.transpose(ims, axes=[2,0,1])[::-1,::,::])

        #mask = np.zeros(rec.shape, dtype=bool)
        mask = create_circular_mask(rec.shape)
        rec = np.swapaxes(rec*mask, 1,2)[::-1]
        del mask
        write_images(rec, filepath, mult)
    if False:
        reg_geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        rec = utils.SIRT_astra(real_image.shape, reg_geo, np.swapaxes(ims, 0,1), 250)
        mask = create_circular_mask(rec.shape)
        rec = rec*mask
        del mask
        #rec = np.swapaxes(rec, 0, 2)
        #rec = np.swapaxes(rec, 1,2)
        #rec = rec[::-1, ::-1]
        rec = sitk.GetImageFromArray(rec)*100
        sitk.WriteImage(rec, os.path.join(outpath, "forcast_matlab_"+name+"_reco-output_sirt.nrrd"))
        del rec

out_rec_meta = ()

def interpol_positions(cs_interpol, Ax, projs, detector_spacing, detector_shape, dist_source_origin, dist_origin_detector, image_spacing):
    no_ims = 0
    coord_systems = []
    thetas = []
    phis = []
    params = []
    for i, (cs, primary, secondary, number) in enumerate(cs_interpol):

        vec = utils.coord_systems2vecs(np.array([cs]), detector_spacing, dist_source_origin, dist_origin_detector, image_spacing)[0]
        cur = np.zeros((3, 3), dtype=float)
        r = utils.rotMat(90, [1,0,0]).dot(utils.rotMat(-90, [0,0,1]))
        cur[1] = r.dot(vec[6:9])
        cur[2] = r.dot(vec[9:12])
        #cur[1] = vec[6:9]
        #cur[2] = vec[9:12]
        print(projs.shape, detector_shape)

        config = dict(cal.default_config)
        data_real = cal.findInitialFeatures(projs[no_ims+2], config)
        config["data_real"] = data_real
        config["real_img"] = projs[no_ims+2]
        config["Ax"] = Ax
        config["it"] = 3
        #cur = cal.correctXY(cur, config)
        #cur = cal.correctZ(cur, config)
        #cur = cal.correctXY(cur, config)
        #cur = cal.correctZ(cur, config)
        #cur = cal.correctXY(cur, config)

        print(cur)
        cur_start, pos = cal.est_position(cur, Ax, projs[no_ims+2])
        plt.figure()
        plt.gray()
        plt.imshow(projs[no_ims+2])
        no_ims += number
        print(cur)
        cur_end, pos_end = cal.est_position(cur, Ax, projs[no_ims-2])
        print(primary, secondary, pos, pos_end)
        cur_end, pos_end = cal.est_position(cur_end, Ax, projs[no_ims-2])
        print(primary, secondary, pos, pos_end)
        plt.figure()
        plt.gray()
        plt.imshow(projs[no_ims-2])
        sims = cal.Projection_Preprocessing(Ax(np.array([cur, cur_start, cur_end])))
        plt.figure()
        plt.gray()
        plt.imshow(sims[:,0])
        plt.figure()
        plt.gray()
        plt.imshow(sims[:,1])
        plt.figure()
        plt.gray()
        plt.imshow(sims[:,2])
        plt.show()
        plt.close()
        #if np.abs(pos_end[0]-pos[0])%180 > np.abs(pos_end[1]-pos[1])%180:
        for p,s,t in zip(np.linspace(pos[0], pos_end[0], number, True), np.linspace(pos[1], pos_end[1], number, True), np.linspace(pos[2], pos_end[2], number, True)):
            params.append(cal.applyRot(cur, p, s, t))
            r = utils.rotMat(p-pos[0], [0,0,1])
            ncs = np.array(cs)
            ncs[:,0] = r.dot(cs[:,0])
            ncs[:,1] = r.dot(cs[:,1])
            ncs[:,2] = r.dot(cs[:,2])
            coord_systems.append(ncs)
            thetas.append(p)
            phis.append(s)
            #phis.append(pos[1])
        #else:
        for i,s in enumerate(np.linspace(pos[1], pos_end[1], number, True)):
            break
            params[i] = cal.applyRot(params[i], 0, s, 0)
            r1 = utils.rotMat(thetas[i], [0,0,1])
            r2 = utils.rotMat(s-pos[1], [1,0,0])
            r = r1.dot(r2)
            ncs = coord_systems[i]
            #cs = np.array(cs)
            ncs[:,0] = r.dot(cs[:,0])
            ncs[:,1] = r.dot(cs[:,1])
            ncs[:,2] = r.dot(cs[:,2])
            coord_systems[i] = ncs
            #coord_systems.append(ncs)
            phis.append(s)
            #thetas.append(pos[0])
    
    coord_systems = np.array(coord_systems)
    thetas = np.array(thetas)
    phis = np.array(phis)
    params = np.array(params)
    return coord_systems, thetas, phis, params

def reg_real_data():
    projs = get_proj_paths()

    #np.seterr(all='raise')

    for name, proj_path, cbct_path, methods in projs:
        
        try:
            ims, ims_un, mas, kvs, angles, coord_systems, sids, sods = read_dicoms(proj_path)
            
            if os.path.exists("Z:\\\\recos"):
                outpath = "Z:\\\\recos"
            elif os.path.exists(r"D:\lumbal_spine_13.10.2020\recos"):
                outpath = r"D:\lumbal_spine_13.10.2020\recos"
            else:
                outpath = r".\recos"

            target_sino = sitk.ReadImage(os.path.join(outpath, "target_sino.nrrd"))
            target_sino = sitk.GetArrayFromImage(target_sino)
            print("target_sino", target_sino.shape)

            #ims = ims[:20]
            #coord_systems = coord_systems[:20]
            #skip = max(1, int(len(ims_un)/500))
            skip = np.zeros(len(ims_un), dtype=bool)
            #skip[200] = True
            skip[::1] = True
            #skip[::max(1, int(len(ims_un)/500))] = True
            random = np.random.default_rng(23)
            #angles_noise = random.normal(loc=0, scale=0.5, size=(len(ims), 3))#*np.pi/180
            angles_noise = random.uniform(low=-2, high=2, size=(len(ims_un),3))
            #angles_noise = np.zeros_like(angles_noise)
            #trans_noise = random.normal(loc=0, scale=20, size=(len(ims), 3))
            min_trans, max_trans = -10, 10
            min_trans, max_trans = -5, 5
            trans_noise = random.uniform(low=min_trans, high=max_trans, size=(len(ims_un),2))
            #zoom_noise = random.uniform(low=0.95, high=1, size=len(ims_un))
            zoom_noise = random.uniform(low=0.98, high=1, size=len(ims_un))

            #skip = 4
            ims = ims[skip]
            ims_un = ims_un[skip]
            coord_systems = coord_systems[skip]
            #angles = angles[skip]
            sids = np.mean(sids[skip])
            sods = np.mean(sods[skip])
            angles_noise = angles_noise[skip]
            trans_noise = trans_noise[skip]
            zoom_noise = zoom_noise[skip]
            angles_noise = np.ones_like(angles_noise)*0
            trans_noise = np.ones_like(trans_noise)*0
            zoom_noise = np.ones_like(zoom_noise)
            #angles_noise[0][0] = -0.05
            #angles_noise[0][1] = -0.166
            #angles_noise[0][2] = -0.393

            coords_from_angles = utils.angles2coord_system(angles)

            origin, size, spacing, image = utils.read_cbct_info(cbct_path)

            detector_shape = np.array((1920,2480))
            detector_mult = int(np.floor(detector_shape[0] / ims_un.shape[1]))
            
            detector_shape = np.array(ims_un.shape[1:])
            detector_spacing = np.array((0.154, 0.154)) * detector_mult

            real_image = utils.fromHU(sitk.GetArrayFromImage(image))
            real_image = np.swapaxes(np.swapaxes(real_image, 0,2), 0,1)[::-1,:,::-1]

            global out_rec_meta
            out_rec_meta = (image.GetOrigin(), image.GetSize(), image.GetSpacing(), real_image.shape)
            del image

            image_spacing = 1.0 / np.min(spacing)
            print(spacing, image_spacing, np.array((1920,2480))/np.array(ims_un[0].shape), detector_mult)

            Ax = utils.Ax_param_asta(real_image.shape, detector_spacing, detector_shape, sods, sids-sods, image_spacing, real_image)

            if coord_systems.shape[1] == 4:
                coord_systems, thetas, phis, params = interpol_positions(coord_systems, Ax, ims, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
                params = params[skip]
            #coord_systems = coord_systems[skip]

            Ax_gen = (real_image.shape, detector_spacing, detector_shape, sods, sids-sods, image_spacing, real_image)
            geo = utils.create_astra_geo_coords(coord_systems, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
            geo_from_angles = utils.create_astra_geo_coords(coords_from_angles, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
            #geo = geo_from_angles
            
            r = utils.rotMat(90, [1,0,0]).dot(utils.rotMat(-90, [0,0,1]))

            if 'arc' in name:
                coord_systems, thetas, phis, params = interpol_positions(coord_systems, Ax, ims, detector_spacing, detector_shape, sods, sids-sods, image_spacing)
                params = params[skip]
                coord_systems = coord_systems[skip]
            elif 'angle' in name or 'both' in name:
                params = np.zeros((len(geo_from_angles['Vectors']), 3, 3), dtype=float)
                params[:,1] = np.array([r.dot(v) for v in geo_from_angles['Vectors'][:, 6:9]])
                params[:,2] = np.array([r.dot(v) for v in geo_from_angles['Vectors'][:, 9:12]])
            else:
                params = np.zeros((len(geo['Vectors']), 3, 3), dtype=float)
                params0 = np.zeros((len(geo['Vectors']), 3, 3), dtype=float)
                params0[:,1,0] = 1
                params0[:,2,1] = 1
                #params[:,0] = coord_systems[:,:,3]
                #params[:,1] = np.array([r.dot(v) for v in geo['Vectors'][:, 6:9]])
                params[:,1] = np.array(geo['Vectors'][:, 6:9])
                #params[:,2] = np.array([r.dot(v) for v in geo['Vectors'][:, 9:12]])
                params[:,2] = np.array(geo['Vectors'][:, 9:12])
            
            print(params[0,1], params[0,2])

            if True:
                for i, (α,β,γ) in enumerate(angles_noise):
                    params[i] = cal.applyRot(params[i], -α, -β, -γ)
            if True:
                for i, (x,y) in enumerate(trans_noise):
                    params[i] = cal.applyTrans(params[i], x, y, 0)
                for i, z in enumerate(zoom_noise):
                    params[i] = cal.applyTrans(params[i], 0, 0, 1-z)

            projs = Ax(params)
            #Ax0201i0 = utils.Ax_param_asta(real_image0201i0.shape, detector_spacing, detector_shape, sods, sids-sods, image_spacing, real_image0201i0)
            #projs0201i0 = Ax0201i0(params0)
            #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/projs.nrrd")
            #sitk.WriteImage(sitk.GetImageFromArray(projs0201i0), "recos/projs0201i0.nrrd")
            #sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(ims,0,1)), "recos/ims.nrrd")

            i0_ims, i0_mas, i0_kvs = i0_data(detector_mult, 60)

            res = np.mean( np.mean(i0_ims, axis=(1,2))[:,np.newaxis,np.newaxis] / i0_ims, axis=0)

            i0s = np.array([i0_est(ims_un[i], projs[:,i])*res for i in range(ims_un.shape[0])])
            i0s = np.mean(i0s, axis=0)
            i0s[i0s==0] = 1e-8
            i0s = np.mean(i0s)
            ims_un = -np.log(ims_un/i0s)

            #sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(-np.log(ims/i0s) ,0,1)))
            #sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_projs_est-input.nrrd"), True)
            #del sino

            #i0s = i0_interpol(i0_ims, i0_mas, np.mean(mas))

            #sino = sitk.GetImageFromArray(cal.Projection_Preprocessing(np.swapaxes(-np.log(ims/i0s) ,0,1)))
            #sitk.WriteImage(sino, os.path.join(outpath, "forcast_"+name+"_projs_int-input.nrrd"), True)
            #del sino

            i0s = np.array([i0_est(ims[i], projs[:,i])*res for i in range(ims.shape[0])])
            i0s = np.mean(i0s, axis=0)
            i0s[i0s==0] = 1e-8
            i0s = np.mean(i0s)
            ims = -np.log(ims/i0s)
            
            #calc_images_matlab("input", ims, real_image, detector_shape, outpath, geo); 
            #calc_images_matlab("genA_trans", ims, real_image, detector_shape, outpath, geo); exit(0)

            config = {"Ax": Ax, "Ax_gen": Ax_gen, "method": 3, "name": name, "real_cbct": real_image, "outpath": outpath, "estimate": False, 
                    "target_sino": target_sino, "threads": mp.cpu_count(), "paralell": True}

            #for method in [3,4,5,0,6]: #-12,-2,-13,-3,20,4,26,31,0,-1
            for method in methods:
                config["name"] = name + str(method)
                config["method"] = method
                config["noise"] = (np.zeros((len(ims),3)), np.array(angles_noise))
                vecs, corrs = reg_and_reco(ims, ims_un, np.array(params), config)
                #iso = (geo['Vectors'][0,0:3]+(sods/sids)*(geo['Vectors'][0,3:6]-geo['Vectors'][0,0:3]))/image_spacing
                #print((params-corrs)[:,0] / image_spacing, origin[0], params[:,0], corrs[:,0]/image_spacing, np.array(real_image.shape)*spacing)
                #print(coord_systems[0,:,3]-iso, geo['Vectors'][0,0:3]/image_spacing-iso, vecs[0,0:3]/image_spacing)
                #print(np.linalg.norm(geo['Vectors'][0,0:3]-geo['Vectors'][0,3:6])/image_spacing, np.linalg.norm(vecs[0,0:3]-vecs[0,3:6])/image_spacing, sids, sods)
                #print(iso)
                #print((vecs[0,0:3]+(sods/sids)*(vecs[0,3:6]-vecs[0,0:3])) /image_spacing)
                #exit()
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