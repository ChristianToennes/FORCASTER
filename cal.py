import numpy as np
#import cv2
import utils
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import sys
import itertools
import time
import threading
import multiprocessing as mp
import queue
from utils import bcolors, applyRot, applyTrans, default_config, filt_conf
from utils import minimize_log as ml
from feature_matching import *
from simple_cal import *
from objectives import *
import cal_bfgs_rot
from skimage.metrics import structural_similarity,normalized_root_mse

import cal_bfgs_both

class OptimizationFailedException(Exception):
    pass

def linsearch(in_cur, axis, config):
    warnings.simplefilter("error")
    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    my = True
    if "my" in config:
        my = config["my"]
    grad_width=(1,25)
    if "grad_width" in config:
        grad_width = config["grad_width"]
    noise=None
    if "angle_noise" in config:
        noise = config["angle_noise"]
    both=False
    if "both" in config:
        both = config["both"]

    objectives = {0:-5, 1:-5, 2:-5}
    if "objectives" in config:
        objectives = config["objectives"]

    use_combined = True
    if "use_combined" in config:
        use_combined = config["use_combined"]
        

    if my:
        points_real, features_real = data_real
        points_real = normalize_points(points_real, real_img)
        real_img = Projection_Preprocessing(real_img)
    else:
        config["GIoldold"] = [None]# = GI(real_img, real_img)
        config["p1"] = [None]
        config["absp1"] = [None]
    #print(grad_width, noise)
    #grad_width = (1,25)
    
    if "binsearch" in config and config["binsearch"] == True:
        make_εs = lambda size, count: np.array([-size/(1.1**i) for i in range(count)] + [0] + [size/(1.1**i) for i in range(count)][::-1])
    else:
        make_εs = lambda size, count: np.hstack((np.linspace(-size, 0, count, False), [0], np.linspace(size, 0, count, False)[::-1] ))
    

    εs = make_εs(*grad_width)
    cur = np.array(in_cur)
    dvec = []
    for ε in εs:
        if axis==0:
            dvec.append(applyRot(cur, ε, 0, 0))
        elif axis==1:
            dvec.append(applyRot(cur, 0, ε, 0))
        elif axis==2:
            dvec.append(applyRot(cur, 0, 0, ε))
    dvec = np.array(dvec)
    projs = Projection_Preprocessing(Ax(dvec))
    if my:
        points = []
        valid = []
        valid2 = []
        for (p,v), proj in [(trackFeatures(projs[:,i], data_real, config), projs[:,i]) for i in range(projs.shape[1])]:
            points.append(normalize_points(p, proj))
            valid.append(v==1)
        combined_valid = valid[0]
        for v in valid:
            combined_valid = np.bitwise_and(combined_valid, v)
        if use_combined and np.count_nonzero(combined_valid) > 5:
            points = [p[combined_valid] for p, v in zip(points, valid)]
            values = np.array([calcPointsObjective(objectives[axis], points, points_real[combined_valid], projs[:,0].shape) for points,v in zip(points,valid)])
        else:
            combined_valid = np.zeros_like(combined_valid)
            points = [p[v] for p, v in zip(points, valid)]
            values = np.array([calcPointsObjective(objectives[axis], points, points_real[v], projs[:,0].shape) for points,v in zip(points,valid)])
    else:
        values = np.array([calcGIObjective(real_img, projs[:,i], 0, None, config) for i in range(projs.shape[1])])

    skip = np.count_nonzero(values<0)
    midpoints = np.argsort(values)[skip:skip+5]
    if len(midpoints) == 0:
        #print(values)
        return cur
    mean_mid = np.median(εs[midpoints])

    if np.max(values) == np.min(values):
        min_ε = εs[len(εs)//2]
        print(" opt failed objectives null ", axis, end=";")
        if False:
            avalues = []
            for comp in [-4, 2, 11, 20, 21, 40, 41]:#[0,1,2,10,11,12,22,32,-1,-2,-3,-4,20,21, 40,41,42]:
                avalues.append( (comp,np.array([calcPointsObjective(comp, points, points_real[combined_valid], img_shape = projs[:,0].shape) for points,v in zip(points,valid)])) )
            fig, axs = plt.subplots(int(np.sqrt(projs.shape[1])), int(np.ceil(projs.shape[1]/int(np.sqrt(projs.shape[1])))), squeeze = True)
            plt.gray()
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(projs[:,i])
            plt.figure()
            plt.gray()
            plt.imshow(real_img)
            plt.figure()
            plt.title(str(axis) + " " + str(my))
            #plt.vlines(min_ε, 0, 1)
            #pv = np.polyval(p, np.linspace(εs[0],εs[-1]))
            #plt.plot(np.linspace(εs[0],εs[-1]), pv, label="p")
            #if p1 is not None:
            #    pv1 = np.polyval(p1, np.linspace(εs[0],εs[-1]))
            #    f = pv1<=np.max(pv)
                #plt.plot(np.linspace(εs[0],εs[-1]), pv1, label="p1")
            for c, v in avalues:
                if (np.max(v)-np.min(v)) != 0:
                    v_norm = (v-np.min(v))/(np.max(v)-np.min(v))
                else:
                    v_norm = v
                if axis==0 and (c in [0, 1, 2, -1, -2, -3, 11, 32, 22, 21, 42, 12, 10, -4] or c in []): # 41 20 40
                    continue
                if axis==1 and (c in [1, 40, 42, 32, -1, -2, -3, 0, 2, 10, 12, 22, 20, -4] or c in []): # 11 21 41
                    continue
                if axis==2 and (c in [0, 1, -1, -2, -3, 32, 42, 11, 12, 22] or c in [21, 10]): # 2 40 41 -4
                    continue
                if (np.max(v)-np.min(v)) == 0:
                    plt.scatter(εs, v, label=str(c))
                else:
                    plt.scatter(εs, v_norm, label=str(c))
                p0 = np.polyfit(εs, v_norm, 4)

                pv = np.polyval(p0, εs)
                f = np.zeros_like(εs, dtype=bool)
                dv = pv-v_norm
                f[np.abs(dv)<3*np.std(dv)] = True
                #p1 = np.polyfit(εs[f], v_norm[f], 4)
                dv = np.argsort(np.abs(pv-v_norm))
                f = np.zeros_like(εs, dtype=bool)
                f[dv[:-5]] = True
                #p2 = np.polyfit(εs[f], v_norm[f], 4)

                def plot_p(p, c):
                    rs = np.real(np.roots(np.polyder(p)))
                    pv = np.polyval(p, np.linspace(εs[0],εs[-1]))
                    l = plt.plot(np.linspace(εs[0],εs[-1]), pv, label=str(c))
                    for r in rs:
                        if r>εs[0] and r<εs[-1]:
                            plt.vlines(r, 0, 1, label=str(c), colors=l[0].get_color())
                
                plot_p(p0, str(c))
                #plot_p(p1, str(c)+"std")
                #plot_p(p2, str(c)+"quant")
            plt.legend()
            plt.ylim(0, 1)
            plt.show()
            plt.close()
        raise OptimizationFailedException("opt failed, all objectives " + str(values[0]) + ", " + str(axis))
    else:
        p = np.polyfit(εs, (values-np.min(values))/(np.max(values)-np.min(values)), 4)
        p1 = None
        r = np.real(np.roots(np.polyder(p)))
        r = [v for v in r if v>εs[0] and v<εs[-1]]
        if len(r) == 0:
            min_ε = mean_mid
        else:
            min_r = np.argmin(np.polyval(p, r))
            min_ε = np.real(r[min_r])


    if False and axis==0:
        avalues = []
        #for comp in [-4, 2, 11, 20, 21, 40, 41]:
        #for comp in [0,1,2,10,11,12,22,32,-1,-2,-3,-4,20,21, 40,41,42]:
        for comp in [-4, -5, -6, -7]:
            if len(valid2) > 0:
                avalues.append( (comp,np.array([calcPointsObjective(comp, points, points_real[v], img_shape = projs[:,0].shape) for points,v in zip(points,valid2)])) )
            elif np.count_nonzero(combined_valid)>0:
                avalues.append( (comp,np.array([calcPointsObjective(comp, points, points_real[combined_valid], img_shape = projs[:,0].shape) for points,v in zip(points,valid)])) )
            else:
                avalues.append( (comp,np.array([calcPointsObjective(comp, points, points_real[v], img_shape = projs[:,0].shape) for points,v in zip(points,valid)])) )
        
        plt.figure()
        plt.title(str(axis) + " " + str(my))
        #plt.vlines(min_ε, 0, 1)
        pv = np.polyval(p, np.linspace(εs[0],εs[-1]))
        #plt.plot(np.linspace(εs[0],εs[-1]), pv, label="p")
        if p1 is not None:
            pv1 = np.polyval(p1, np.linspace(εs[0],εs[-1]))
            f = pv1<=np.max(pv)
            #plt.plot(np.linspace(εs[0],εs[-1]), pv1, label="p1")
        for c, v in avalues:
            v_norm = (v-np.min(v))/(np.max(v)-np.min(v))
            if axis==0 and (c in [0,1,2,-1,-2,-3,11,10,12,22,32,20] or c in [41,40]): # 42 41 20 40
                continue
            if axis==1 and (c in [1, 40, 32, -1, -2, -3, 0, 2, 10, 12, 22, 20] or c in []): # 42 11 21 41
                continue
            if axis==2 and (c in [0, 1, -1, -2, -3, 32, 11, 12, 22] or c in [21, 10]): # 42 2 40 41 -4
                continue
            if (np.max(v)-np.min(v)) == 0:
                plt.scatter(εs, v, label=str(c))
            else:
                plt.scatter(εs, v_norm, label=str(c))
            p0 = np.polyfit(εs, v_norm, 4)

            pv = np.polyval(p0, εs)
            f = np.zeros_like(εs, dtype=bool)
            dv = pv-v_norm
            f[np.abs(dv)<3*np.std(dv)] = True
            p1 = np.polyfit(εs[f], v_norm[f], 4)
            dv = np.argsort(np.abs(pv-v_norm))
            f = np.zeros_like(εs, dtype=bool)
            f[dv[:-5]] = True
            p2 = np.polyfit(εs[f], v_norm[f], 4)

            def plot_p(p, c):
                rs = np.real(np.roots(np.polyder(p)))
                pv = np.polyval(p, np.linspace(εs[0],εs[-1]))
                l = plt.plot(np.linspace(εs[0],εs[-1]), pv, label=str(c))
                for r in rs:
                    if r>εs[0] and r<εs[-1]:
                        plt.vlines(r, 0, 1, label=str(c), colors=l[0].get_color())
            
            plot_p(p0, str(c))
            #plot_p(p1, str(c)+"std")
            #plot_p(p2, str(c)+"quant")
        plt.legend()
        plt.ylim(0, 1)
        plt.show()
        plt.close()
    

    if axis==0:
        cur = applyRot(cur, min_ε, 0, 0)
    if axis==1:
        cur = applyRot(cur, 0, min_ε, 0)
    if axis==2:
        cur = applyRot(cur, 0, 0, min_ε)
    if False and noise is not None:
        print("{}{}{} {: .3f} {: .3f}".format(bcolors.BLUE, axis, bcolors.END, noise[axis], min_ε), end=": ")
        if np.abs(noise[axis]) < np.abs(noise[axis]-min_ε)-0.1:
            print("{}{: .3f}{}".format(bcolors.RED, noise[axis]-min_ε, bcolors.END), end=", ")
        elif np.abs(noise[axis]) > np.abs(noise[axis]-min_ε):
            print("{}{: .3f}{}".format(bcolors.GREEN, noise[axis]-min_ε, bcolors.END), end=", ")
        else:
            print("{}{: .3f}{}".format(bcolors.YELLOW, noise[axis]-min_ε, bcolors.END), end=", ")
        #noise[axis] -= min_ε
    if noise is not None:
        noise[axis] -= min_ε

    if both:
        return cur, min_ε
    return cur


def linsearch2d(in_cur, axis, config):
    warnings.simplefilter("error")
    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    my = True
    if "my" in config:
        my = config["my"]
    grad_width=(1,25)
    if "grad_width" in config:
        grad_width = config["grad_width"]
    noise=None
    if "angle_noise" in config:
        noise = config["angle_noise"]
    both=False
    if "both" in config:
        both = config["both"]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)
    if not my:
        config["GIoldold"] = [None]# = GI(real_img, real_img)
        config["p1"] = [None]
        config["absp1"] = [None]
    #print(grad_width, noise)
    #grad_width = (1,25)
    
    if "binsearch" in config and config["binsearch"] == True:
        make_εs = lambda size, count: np.array([-size/(1.1**i) for i in range(count)] + [0] + [size/(1.1**i) for i in range(count)][::-1])
    else:
        make_εs = lambda size, count: np.hstack((np.linspace(-size, 0, count, False), [0], np.linspace(size, 0, count, False)[::-1] ))
    

    εs = make_εs(*grad_width)
    cur = np.array(in_cur)
    dvec = []
    for ε1 in εs:
        for ε2 in εs:
            dvec.append(applyRot(cur, ε1, ε2, 0))

    dvec = np.array(dvec)
    projs = Projection_Preprocessing(Ax(dvec))
    if my:
        points = []
        valid = []
        for (p,v), proj in [(trackFeatures(projs[:,i], data_real, config), projs[:,i]) for i in range(projs.shape[1])]:
            points.append(normalize_points(p, proj))
            valid.append(v==1)
        combined_valid = valid[0]
        for v in valid:
            combined_valid = np.bitwise_and(combined_valid, v)
        points = [p[v] for p, v in zip(points, valid)]
        
        values = np.array([calcPointsObjective(0, points, points_real[v])+calcPointsObjective(1, points, points_real[v]) for points,v in zip(points,valid)])
        values = values.reshape(( len(εs),len(εs) ))
        skip = np.count_nonzero(values<0)
        midpoints = np.array(np.unravel_index(np.argsort(values,axis=None), values.shape))[skip:skip+5]
        if len(midpoints) == 0:
            #print(values)
            return cur
        mean_mid = np.median(εs[midpoints], axis=0)
        min_ε = mean_mid
    else:
        values = np.array([calcGIObjective(real_img, projs[:,i], 0, None, config) for i in range(projs.shape[1])])
        skip = np.count_nonzero(values<0)
        midpoints = np.array(np.unravel_index(np.argsort(values,axis=None), values.shape))[skip:skip+5]
        mean_mid = np.mean(εs[midpoints], axis=0)
        min_ε = mean_mid

    cur = applyRot(cur, min_ε[0], min_ε[1], 0)
    if both:
        return cur, min_ε
    return cur


def binsearch(in_cur, axis, config):
    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    my = True
    if "my" in config:
        my = config["my"]
    grad_width=(1,25)
    if "grad_width" in config:
        grad_width = config["grad_width"]
    noise=None
    if "angle_noise" in config:
        noise = config["angle_noise"]
    both=False
    if "both" in config:
        both = config["both"]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)
    if not my:
        config["GIoldold"] = [None]# = GI(real_img, real_img)
        config["p1"] = [None]
        config["absp1"] = [None]

    make_εs = lambda size, count: np.array([-size/(1.1**i) for i in range(count)] + [0] + [size/(1.1**i) for i in range(count)][::-1])
    εs = make_εs(*grad_width)
    change = 0
    selected_εs = []
    cur = np.array(in_cur)
    failed_count = grad_width[0]
    osci_count = 0
    min_ε = 0
    for it in range(5):
        #print(εs, change)
        dvec = []
        for ε in εs:
            if axis==0:
                dvec.append(applyRot(cur, ε, 0, 0))
            elif axis==1:
                dvec.append(applyRot(cur, 0, ε, 0))
            elif axis==2:
                dvec.append(applyRot(cur, 0, 0, ε))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        if my:
            values = np.array([1-calcMyObjective(axis, projs[:,i], config) for i in range(projs.shape[1])])
            #values1 = np.array([calcObjective(data_real, real_img, projs[:,i], {}, GIoldold) for i in range(projs.shape[1])])
            p = np.polyfit(εs[values>=0], values[values>=0], 2)
            #p1 = np.polyfit(εs, values1, 2)
        else:
            values = np.array([1-calcGIObjective(real_img, projs[:,i], 0, None, config) for i in range(projs.shape[1])])
            p = np.polyfit(εs, values, 2)
        #plt.figure()
        #plt.plot(np.linspace(1.2*εs[0],1.2*εs[-1]), np.polyval(p, np.linspace(1.2*εs[0],1.2*εs[-1])))
        #plt.scatter(εs, values)
        #plt.figure()
        #plt.plot(np.linspace(1.2*εs[0],1.2*εs[-1]), np.polyval(p1, np.linspace(1.2*εs[0],1.2*εs[-1])))
        #plt.scatter(εs, values1)
        #plt.show()
        #plt.close()
    
        mins = np.roots(np.polyder(p))
        ε_pos = np.argmin(np.polyval(p, mins))
        if len(selected_εs)>10 and np.abs(np.mean(selected_εs[-8:]) < 0.000001):
        #if (mins[ε_pos]>0) != (min_ε>0):
            osci_count += 1
            #min_ε = (mins[ε_pos]+min_ε)*0.5
        else:
            osci_count = 0
        if osci_count > 8:
            min_ε = np.mean(selected_εs[-8:])
        else:
            min_ε = mins[ε_pos]
        change += min_ε
        nearest_index = 0
        for i, ε in enumerate(εs):
            if np.abs(ε-min_ε) < np.abs(εs[nearest_index]-min_ε):
                nearest_index = i
        #print(nearest_index, εs[nearest_index], min_ε)
        if nearest_index > 0 and nearest_index < len(εs)-1:
            εs = make_εs(εs[nearest_index+1]-εs[nearest_index-1], grad_width[1])
        else:
            if nearest_index == 0:
                εs = make_εs(2*(εs[nearest_index+1]-εs[nearest_index]), grad_width[1])
            else:
                εs = make_εs(2*(εs[nearest_index]-εs[nearest_index-1]), grad_width[1])
        if np.abs(min_ε) <= 0.001*grad_width[0]:
            print("small", end=" ")
            if axis==0:
                cur = applyRot(cur, min_ε, 0, 0)
                #cur = applyTrans(cur, med_x, 0, 0)
            if axis==1:
                cur = applyRot(cur, 0, min_ε, 0)
                #cur = applyTrans(cur, 0, med_y, 0)
            if axis==2:
                cur = applyRot(cur, 0, 0, min_ε)    
                #cur = applyTrans(cur, 0, 0, scale)
            break
        if np.abs(change) > failed_count:
            #print("reset", axis, change, end=", ", flush=True)
            print("reset", end=" ")
            failed_count += grad_width[0]
            osci_count = 0
            cur = np.array(in_cur)
            if np.abs(change) < grad_width[0]*1:
                #εs = make_εs(*grad_width)
                #min_ε = i*np.random.random(1)[0]-0.5*i
                min_ε = εs[np.argmin(values)]
            else:
                #εs = make_εs(grad_width[0]*(2**(failed_count/grad_width[0])), grad_width[0]+6*int(failed_count/grad_width[0]))
                #min_ε = 0
                min_ε = εs[np.argmin(values)]
            min_ε = np.random.uniform(εs[1], εs[-2])
            change = min_ε
        if failed_count > grad_width[0]*6:
            #print("failed", axis, grad_width, end=", ", flush=True)
            print("failed", end=" ")
            cur = np.array(in_cur)
            break
        selected_εs.append(min_ε)
        if osci_count > 10:
            #print("osci", axis, end=",")
            break
        if axis==0:
            cur = applyRot(cur, min_ε, 0, 0)
            #cur = applyTrans(cur, med_x, 0, 0)
        if axis==1:
            cur = applyRot(cur, 0, min_ε, 0)
            #cur = applyTrans(cur, 0, med_y, 0)
        if axis==2:
            cur = applyRot(cur, 0, 0, min_ε)    
            #cur = applyTrans(cur, 0, 0, scale)
    if False and noise is not None:
        print("{}{}{} {: .3f} {: .3f}".format(bcolors.BLUE, axis, bcolors.END, noise[axis], change), end=": ")

        if np.abs(noise[axis]) < np.abs(noise[axis]-change)-0.1:
            print("{}{: .3f}{}".format(bcolors.RED, noise[axis]-change, bcolors.END), end=", ")
        elif np.abs(noise[axis]) > np.abs(noise[axis]-change):
            print("{}{: .3f}{}".format(bcolors.GREEN, noise[axis]-change, bcolors.END), end=", ")
        else:
            print("{}{: .3f}{}".format(bcolors.YELLOW, noise[axis]-change, bcolors.END), end=", ")
        noise[axis] -= change
    if both:
        return cur, change
    return cur

def log_error(cur, config):
    return
    Ax = config["Ax"]
    proj = Ax(np.array([cur]))[:,0]
    config["log_queue"].put( ("logs/"+config["name"], (structural_similarity(config["target_sino"], proj), normalized_root_mse(config["target_sino"], proj))) )

def roughRegistration(in_cur, reg_config, c):
    cur = np.array(in_cur)
    config = dict(default_config)
    config.update(reg_config)

    config["Ax"] = config["Ax_small"]
    config["real_img"] = config["real_img_small"]

    if c==-19:
        config["opti_method"] = "Newton-CG"
        return bfgs(cur, config, 1)
    if c==-18:
        config["opti_method"] = "TNC"
        return bfgs(cur, config, 1)
    if c==-17:
        config["opti_method"] = "trust-exact"
        return bfgs(cur, config, 1)
    if c==-16:
        config["opti_method"] = "trust-krylov"
        return bfgs(cur, config, 1)
    if c==-15:
        config["opti_method"] = "trust-ncg"
        return bfgs(cur, config, 1)
    if c==-14:
        config["opti_method"] = "dogleg"
        return bfgs(cur, config, 1)
    if c==-13:
        config["opti_method"] = "SLSQP"
        return bfgs(cur, config, 1)
    if c==-12:
        config["opti_method"] = "COBYLA"
        return bfgs(cur, config, 1)
    if c==-7:
        config["opti_method"] = "trust-exact"
        return bfgs(cur, config, 0)
    if c==-6:
        config["opti_method"] = "trust-krylov"
        return bfgs(cur, config, 0)
    if c==-5:
        config["opti_method"] = "trust-ncg"
        return bfgs(cur, config, 0)
    if c==-4:
        config["opti_method"] = "dogleg"
        return bfgs(cur, config, 0)
    if c==-3:
        config["opti_method"] = "SLSQP"
        return bfgs(cur, config, 0)
    if c==-2:
        config["opti_method"] = "COBYLA"
        return bfgs(cur, config, 0)
    if c==-1:
        return bfgs(cur, reg_config, 0)
    if c==0:
        return bfgs(cur, reg_config, 1)
    if c==1 or c==2:
        return bfgs_trans(cur, reg_config, c)
    if c<=-30:
        return cal_bfgs_rot.bfgs(cur, reg_config, c)
    if c<=-20:
        return bfgs_trans_all(cur, reg_config, c)

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    #print("rough")
    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)
    
    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]
    
    est_data = config["est_data"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)
    
    #plt.figure()
    #plt.imshow(real_img)
    #plt.figure()
    #plt.imshow(proj_img)
    #plt.show()
    #plt.close()

    def grad_img(img):
        i = img[1:]-img[:-1]
        i = i[:,1:]-i[:,:-1]
        return i

    def debug_projs(cur):
        plt.figure()
        plt.title("real")
        plt.imshow(grad_img(real_img))
        plt.figure()
        plt.title("cur_proj")
        cur_proj = Projection_Preprocessing(Ax(np.array([cur]))[:,0])
        plt.imshow(grad_img(cur_proj))
        plt.figure()
        plt.title("diff_prev")
        plt.imshow(grad_img(real_img)-grad_img(proj_img))
        plt.figure()
        plt.title("diff_cur")
        plt.imshow(grad_img(real_img)-grad_img(cur_proj))
        plt.figure()
        plt.title("diff_proj")
        plt.imshow(grad_img(proj_img)-grad_img(cur_proj))
        plt.show()
        plt.close()

    if c==3: # 3
        config["it"] = 1
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==4: # 4
        correctFlip(cur, config)
        config["it"] = 5
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c == 5:
        config["it"] = 5
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c == 6:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c == 7:
        config["it"] = 1
        config["mean"] = False
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==8:
        config["it"] = 3
        config["mean"] = True
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==9:
        config["it"] = 1
        config["mean"] = True
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==10:
        config["it"] = 2
        config["mean"] = True
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==11:
        config["it"] = 2
        config["mean"] = False
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==12:
        config["it"] = 1
        config["mean"] = False
        for _ in range(3):
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
    elif c==13:
        config["it"] = 1
        config["mean"] = False
        for _ in range(2):
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
    elif c==15:
        config["my"] = False
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        for grad_width in [(1.5,25), (0.5,25)]:
            #print()
            for _ in range(2):
                config["grad_width"]=grad_width
                cur = binsearch(cur, 0, config)
                cur = binsearch(cur, 1, config)
                cur = binsearch(cur, 2, config)
                cur = correctXY(cur, config)
                cur = correctZ(cur, config)
    elif c==16:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        for grad_width in [(2,25), (1.5,25), (1,25), (0.75,25), (0.75,25), (0.5,25), (0.5,25), (0.25,25)]:
            #print()
            config["grad_width"]=grad_width
            cur = linsearch(cur, 0, config)
            #cur = binsearch(cur, 1, config)
            #cur = binsearch(cur, 2, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
        for grad_width in [(2,25), (1.5,25), (1,25), (0.75,25), (0.75,25), (0.5,25), (0.5,25), (0.25,25)]:
            #print()
            config["grad_width"]=grad_width
            #cur = binsearch(cur, 0, config)
            cur = linsearch(cur, 1, config)
            #cur = binsearch(cur, 2, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
        for grad_width in [(2,25), (1.5,25), (1,25), (0.75,25), (0.75,25), (0.5,25), (0.5,25), (0.25,25)]:
            #print()
            config["grad_width"]=grad_width
            #cur = binsearch(cur, 0, config)
            #cur = binsearch(cur, 1, config)
            cur = linsearch(cur, 2, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)

        for grad_width in [(2,25), (1.5,25), (1,25), (0.75,25), (0.75,25), (0.5,25), (0.5,25), (0.25,25)]:
            #print()
            config["grad_width"]=grad_width
            cur = linsearch(cur, 0, config)
            #cur = binsearch(cur, 1, config)
            #cur = binsearch(cur, 2, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
        for grad_width in [(2,25), (1.5,25), (1,25), (0.75,25), (0.75,25), (0.5,25), (0.5,25), (0.25,25)]:
            #print()
            config["grad_width"]=grad_width
            #cur = binsearch(cur, 0, config)
            cur = linsearch(cur, 1, config)
            #cur = binsearch(cur, 2, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
        for grad_width in [(2,25), (1.5,25), (1,25), (0.75,25), (0.75,25), (0.5,25), (0.5,25), (0.25,25)]:
            #print()
            config["grad_width"]=grad_width
            #cur = binsearch(cur, 0, config)
            #cur = binsearch(cur, 1, config)
            cur = linsearch(cur, 2, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
    elif c==19:
        pass
    elif c==20:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        for grad_width in [(2.5,15), (0.5,15)]:
            #print()
            for _ in range(2):
                config["grad_width"]=grad_width
                cur = linsearch(cur, 0, config)
                cur = linsearch(cur, 1, config)
                cur = linsearch(cur, 2, config)
                cur = correctXY(cur, config)
                cur = correctZ(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==21:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["binsearch"] = True
        for grad_width in [(2.5,15), (0.5,15)]:
            #print()
            for _ in range(2):
                config["grad_width"]=grad_width
                cur = linsearch(cur, 0, config)
                cur = linsearch(cur, 1, config)
                cur = linsearch(cur, 2, config)
                cur = correctXY(cur, config)
                cur = correctZ(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==22:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["binsearch"] = True
        for grad_width in [(2.5,15), (0.5,15)]:
            #print()
            for _ in range(3):
                config["grad_width"]=grad_width
                cur = linsearch(cur, 0, config)
                cur = linsearch(cur, 1, config)
                cur = linsearch(cur, 2, config)
                cur = correctXY(cur, config)
                cur = correctZ(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==23:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        for grad_width in [(2.5,15), (0.5,15)]:
            #print()
            for _ in range(2):
                config["grad_width"]=grad_width
                config["binsearch"] = False
                cur = linsearch(cur, 0, config)
                config["binsearch"] = False
                cur = linsearch(cur, 1, config)
                config["binsearch"] = True
                cur = linsearch(cur, 2, config)
                cur = correctXY(cur, config)
                cur = correctZ(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==24:
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        for grad_width in [(1.5,15), (1,15), (0.5,15), (0.25,15)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            if False:
                for axis in [0,1,2]:
                    print("{}{}{}".format(bcolors.BLUE, axis, bcolors.END), end=": ")
                    bcolors.print_val(np.mean(config["angle_noise"][axis]))
                    bcolors.print_val(np.std(config["angle_noise"][axis]))
                    bcolors.print_val(np.min(config["angle_noise"][axis]))
                    bcolors.print_val(np.quantile(config["angle_noise"][axis], 0.25))
                    bcolors.print_val(np.median(config["angle_noise"][axis]))
                    bcolors.print_val(np.quantile(config["angle_noise"][axis], 0.75))
                    bcolors.print_val(np.max(config["angle_noise"][axis]))
                    print()
    elif c==25: # bad

        cur = correctFlip(cur, config)

        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        grad_widths = [(0.25,7), (1.5,7)]
        failed_counter = 0
        while len(grad_widths) > 0:
            grad_width = grad_widths.pop()
            config["grad_width"]=grad_width
            d0 = 0
            d1 = 0
            d2 = 0
            failed = False
            try:
                _cur, d2 = linsearch(cur, 2, config)
            except OptimizationFailedException as e:
                failed = True
            try:
                _cur, d0 = linsearch(cur, 0, config)
            except OptimizationFailedException as e:
                failed = True
            try:
                _cur, d1 = linsearch(cur, 1, config)
            except OptimizationFailedException as e:
                failed = True
            if failed:
                if failed_counter < 3:
                    grad_widths.append(grad_width)
                    failed_counter += 1
                    print("failed restart", failed_counter)
                else:
                    failed_counter = 0
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            if False:
                for axis in [0,1,2]:
                    print("{}{}{}".format(bcolors.BLUE, axis, bcolors.END), end=": ")
                    bcolors.print_val(np.mean(config["angle_noise"][axis]))
                    bcolors.print_val(np.std(config["angle_noise"][axis]))
                    bcolors.print_val(np.min(config["angle_noise"][axis]))
                    bcolors.print_val(np.quantile(config["angle_noise"][axis], 0.25))
                    bcolors.print_val(np.median(config["angle_noise"][axis]))
                    bcolors.print_val(np.quantile(config["angle_noise"][axis], 0.75))
                    bcolors.print_val(np.max(config["angle_noise"][axis]))
                    print()
    elif c==26:
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.5,9), (0.25,9), (0.25,9), (0.1,9)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
    elif c==27:
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        config["objectives"] = {0: -5, 1: -5, 2: -5}
        for grad_width in [(4,20), (2.5,15), (1.5,10), (1,7), (0.5,7), (0.25,7), (0.1, 7)]:
            config["grad_width"]=grad_width
            if grad_width[0] > 1.5:
                config["objectives"] = {0: -9, 1: -9, 2: -9}
                config["use_combined"] = False
            else:
                config["objectives"] = {0: -5, 1: -5, 2: -5}
                config["use_combined"] = True
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
    elif c==28: # bad
        cur = correctFlip(cur, config)
        config["it"] = 2
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 2
        config["both"] = True
        config["objectives"] = {0: -8, 1: -8, 2: -8}
        for grad_width in [(1.5,5), (1,5), (0.5,5), (0.25,5)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
    elif c==29:
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)

    elif c==30: # bad
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.5,9), (0.25,9), (0.25,9), (0.1,9)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)

    elif c==31: # bad

        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 1
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    
    elif c==32:
        
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    
    elif c==33:
        
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "33 QUT-AF my " + config["name"]
        log_error(cur, config)
        cur = correctFlip(cur, config)
        config["it"] = 3
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)

        config["it"] = 2
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        ml("33 QUT-AF my", starttime, res)

    elif c==34:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 2
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        ml("34 QUT-AF my reduced noise", starttime, res)

    elif c==35:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "35 QUT-AF my " + config["name"]
        log_error(cur, config)
        cur = correctFlip(cur, config)
        config["it"] = 3
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)

        config["it"] = 2
        config["both"] = True
        config["objectives"] = {0: -4, 1: -3, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        ml("35 QUT-AF my", starttime, res)

    elif c==41:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["my"] = False
        cur = correctFlip(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        config["both"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.5,9), (0.25,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            cur = correctXY(cur, config)
            
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        ml("41 QUT-AF ngi reduced noise", starttime, res)
    elif c==42:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["my"] = False
        config["name"] = "42 QUT-AF ngi " + config["name"]
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)

        config["it"] = 3
        config["both"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        ml("42 QUT-AF ngi", starttime, res)

    elif c==60:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "60 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        ml("60 EST-QUT-AF my", starttime, res)

    elif c==60.5:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "60 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 1
        cur = correctFlip(cur, config)
        cur = correctZ(cur, config)
        cur2 = np.array(cur)
        try:
            cur = correctRotZ(cur, config)
            #print("rz", cur)
            log_error(cur, config)
        except Exception as e:
            print(e)
            cur = cur2
        cur = correctXY(cur, config)

        ml("60.5 EST-QUT-AF my", starttime, res)

    elif c==61:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "61 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]
        
        config["it"] = 3
        log_error(cur, config)
        cur = correctFlip(cur, config)
        cur = correctZ(cur, config)
        cur2 = np.array(cur)
        try:
            cur = correctRotZ(cur, config)
            #print("rz", cur)
            log_error(cur, config)
        except Exception as e:
            print(e)
            cur = cur2
        #print("f", cur)
        log_error(cur, config)
        cur2 = np.array(cur)
        for i in range(1):
            cur = correctXY(cur, config)
            #print("xy", cur)
            log_error(cur, config)
            cur = correctZ(cur, config)
            #print("z", cur)
            log_error(cur, config)
            cur2 = np.array(cur)
            try:
                cur = correctRotZ(cur, config)
                #print("rz", cur)
                log_error(cur, config)
            except Exception as e:
                print(e)
                cur = cur2
            cur = correctXY(cur, config)
            #print("xy", cur)
            log_error(cur, config)

        ml("61 EST-QUT-AF my", starttime, res)

    elif c==62:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["my"] = False
        config["name"] = "62 QUT-AF ngi " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]
        #print("e", cur)

        config["it"] = 3
        log_error(cur, config)
        cur = correctFlip(cur, config)
        cur = correctZ(cur, config)
        cur = correctRotZ(cur, config)
        #print("f", cur)
        log_error(cur, config)
        cur2 = np.array(cur)
        for i in range(1):
            cur = correctXY(cur, config)
            #print("xy", cur)
            log_error(cur, config)
            cur = correctZ(cur, config)
            #print("z", cur)
            cur2 = np.array(cur)
            try:
                cur = correctRotZ(cur, config)
                #print("rz", cur)
                log_error(cur, config)
            except Exception as e:
                print(e)
                cur = cur2
            log_error(cur, config)
            cur = correctXY(cur, config)
            #print("xy", cur)
            log_error(cur, config)

        #config["Ax"] = config["Ax_big"]
        #config["real_img"] = config["real_img_big"]

        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)

        config["it"] = 3
        config["both"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            #cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        #cur = correctRotZ(cur, config)
        #log_error(cur, config)
        ml("62 EST-QUT-AF ngi", starttime, res)

    elif c==62.5:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "62.5 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 1
        cur = correctFlip(cur, config)
        cur = correctZ(cur, config)
        cur2 = np.array(cur)
        try:
            cur = correctRotZ(cur, config)
            #print("rz", cur)
            log_error(cur, config)
        except Exception as e:
            print(e)
            cur = cur2
        cur = correctXY(cur, config)

        log_error(cur, config)
        cur = correctFlip(cur, config)
        config["it"] = 3
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)

        config["it"] = 2
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, d2)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)

        ml("62.5 EST-QUT-AF my", starttime, res)
    
    elif c==621:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["my"] = False
        config["name"] = "62 QUT-AF ngi " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 1
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            _cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        #cur = correctRotZ(cur, config)
        #log_error(cur, config)
        ml("62 EST-QUT-AF ngi", starttime, res)

    elif c==63:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "63 QUT-AF my " + config["name"]
        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 2
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        for i in range(2):
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctZ(cur, config)
            log_error(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            #_cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctRotZ(cur, config)
        log_error(cur, config)
        ml("63 EST-QUT-AF my", starttime, res)

    elif c==630:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "63 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 2
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        for i in range(2):
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctZ(cur, config)
            log_error(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        config["objectives"] = {0: -6, 1: -6, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            #_cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctRotZ(cur, config)
        log_error(cur, config)
        ml("63 EST-QUT-AF my", starttime, res)

    elif c==631:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "63 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 2
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        for i in range(2):
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctZ(cur, config)
            log_error(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        config["objectives"] = {0: -4, 1: -3, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            #_cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctRotZ(cur, config)
        log_error(cur, config)
        ml("63 EST-QUT-AF my", starttime, res)

    elif c==632:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "63 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 1
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        config["objectives"] = {0: -4, 1: -3, 2: -6}
        config["use_combined"] = True
        for grad_width in [(2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            #_cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctRotZ(cur, config)
        log_error(cur, config)
        ml("63 EST-QUT-AF my", starttime, res)

    elif c==64:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["my"] = False
        config["name"] = "64 QUT-AF ngi " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 2
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        for i in range(2):
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctZ(cur, config)
            log_error(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        config["my"] = False
        for grad_width in [(3, 15), (2.5, 15), (2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            #_cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctRotZ(cur, config)
        log_error(cur, config)
        ml("64 EST-QUT-AF ngi", starttime, res)

    elif c==65:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "65 QUT-AF my " + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 2
        log_error(cur, config)
        cur = correctFlip(cur, config)
        log_error(cur, config)
        for i in range(2):
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctZ(cur, config)
            log_error(cur, config)
            cur = correctXY(cur, config)
            log_error(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)

        config["it"] = 1
        config["both"] = True
        config["objectives"] = {0: -4, 1: -3, 2: -6}
        config["use_combined"] = True
        for grad_width in [(3,15), (2.5,15), (2,9),(1.5,9), (1,9), (0.5,9), (0.25,9), (0.1,9)]:
            res["nit"] += 1
            res["nfev"] += grad_width[1] * 2 * 3
            res["njev"] += 3
            config["grad_width"]=grad_width
            #_cur, d2 = linsearch(cur, 2, config)
            _cur, d0 = linsearch(cur, 0, config)
            _cur, d1 = linsearch(cur, 1, config)
            cur = applyRot(cur, d0, d1, 0)
            log_error(cur, config)
            cur = correctXY(cur, config)
            cur = correctZ(cur, config)
            cur = correctXY(cur, config)
            cur = correctRotZ(cur, config)
            log_error(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctZ(cur, config)
        log_error(cur, config)
        cur = correctXY(cur, config)
        log_error(cur, config)
        cur = correctRotZ(cur, config)
        log_error(cur, config)
        ml("65 EST-QUT-AF my", starttime, res)

    elif c==70:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "70 EST-QUT-AF my" + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        config["my"] = False

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 1
        cur = correctFlip(cur, config)
        cur = correctZ(cur, config)
        cur2 = np.array(cur)
        try:
            cur = correctRotZ(cur, config)
            #print("rz", cur)
            log_error(cur, config)
        except Exception as e:
            print(e)
            cur = cur2
        cur = correctXY(cur, config)

        cur = cal_bfgs_both.bfgs(cur, config, -70)

        ml("70 EST-QUT-AF my", starttime, res)
    
    elif c==71:
        starttime = time.perf_counter()
        res = {"success": True, "nit": 0, "nfev": 0, "njev": 0, "nhev": 0}
        config["name"] = "70 EST-QUT-AF my" + config["name"]

        cur0 = np.zeros((3, 3), dtype=float)
        cur0[1,0] = 1
        cur0[2,1] = 1

        config["my"] = True

        #if("est_data" in config):
        #    est_data = config["est_data"]
        #else:
        #    est_data = simulate_est_data(cur0, Ax)
        #    config["est_data"] = est_data
        cur, _rots = est_position(cur0, Ax, [real_img], est_data)
        cur = cur[0]

        config["it"] = 1
        cur = correctFlip(cur, config)
        cur = correctZ(cur, config)
        cur2 = np.array(cur)
        try:
            cur = correctRotZ(cur, config)
            #print("rz", cur)
            log_error(cur, config)
        except Exception as e:
            print(e)
            cur = cur2
        cur = correctXY(cur, config)

        cur = cal_bfgs_both.bfgs(cur, config, -72)

        ml("70 EST-QUT-AF my", starttime, res)

    return cur

def bfgs(cur, reg_config, c):
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c==1

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)

    if 'opti_method' not in config:
        config['opti_method'] = 'L-BFGS-B'
    #else:
    #    print(config['opti_method'],end=';',flush=True)
    
    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    data_real = config["data_real"]
    points_real = normalize_points(data_real[0], real_img)
    if config["my"]:
        def f(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]

            (p,v), proj = trackFeatures(proj, data_real, config), proj
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            ret = 0
            for axis, mult in [(0,1),(1,1),(2,1)]:
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    ret += 50
                else:
                    ret += obj
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyRot(cur_x, eps[0], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            (p,v), proj = trackFeatures(projs[:,0], data_real, config), projs[:,0]
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            cur_obj = []
            for axis, mult in [(0,1),(1,1),(2,1)]:
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    cur_obj.append(50)
                else:
                    cur_obj.append(obj)

            ret = [0,0,0]
            for i in range(1, projs.shape[1]):
                proj = projs[:,i]
                (p,v), proj = trackFeatures(proj, data_real, config), proj
                points = normalize_points(p, proj)
                valid = v==1
                points = points[valid]
                part = 0
                axis, mult = [(0,1),(1,1),(2,1)][i-1]
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    obj = 50

                ret[i-1] = (obj-cur_obj[i-1])*eps[i-1]

            return ret
        
        def hessf(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyRot(cur_x, eps[0], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]))
            dvec.append(applyRot(cur_x, eps[0]*2, 0, 0))
            dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, 0, eps[1]*2, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]*2))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            (p,v), proj = trackFeatures(projs[:,0], data_real, config), projs[:,0]
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            cur_obj = []
            for axis, mult in [(0,1),(1,1),(2,1)]:
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    cur_obj.append(50)
                else:
                    cur_obj.append(obj)
            
            cur_obj = np.array(cur_obj)

            objs = np.zeros((9,3), dtype=cur_obj.dtype)
            for i in range(1, projs.shape[1]):
                proj = projs[:,i]
                (p,v), proj = trackFeatures(proj, data_real, config), proj
                points = normalize_points(p, proj)
                valid = v==1
                points = points[valid]
                part = 0
                axis = [(0,),(1,),(2,),(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)][i-1]
                for ax in axis:
                    obj = calcPointsObjective(axis, points, points_real[valid])
                    if obj==-1:
                        obj = 50
                    objs[i-1][ax] = obj
                
            ret = np.zeros((3,3), dtype=cur_obj.dtype)
            ret[0,0] = np.sum( (objs[3] - objs[0] - objs[0] + cur_obj) / (eps[0]*eps[0]) )
            ret[0,1] = np.sum( (objs[4] - objs[0] - objs[1] + cur_obj) / (eps[0]*eps[1]) )
            ret[0,2] = np.sum( (objs[5] - objs[0] - objs[2] + cur_obj) / (eps[0]*eps[0]) )
            #ret[1,0] = np.sum( (objs[4] - objs[1] - objs[0] + cur_obj) / (eps[1]*eps[0]) )
            ret[1,0] = ret[0,1]
            ret[1,1] = np.sum( (objs[6] - objs[1] - objs[1] + cur_obj) / (eps[1]*eps[1]) )
            ret[1,2] = np.sum( (objs[7] - objs[1] - objs[2] + cur_obj) / (eps[1]*eps[2]) )
            #ret[2,0] = np.sum( (objs[5] - objs[2] - objs[0] + cur_obj) / (eps[2]*eps[0]) )
            ret[2,0] = ret[0,2]
            #ret[2,1] = np.sum( (objs[7] - objs[2] - objs[1] + cur_obj) / (eps[2]*eps[1]) )
            ret[2,1] = ret[1,2]
            ret[2,2] = np.sum( (objs[8] - objs[2] - objs[2] + cur_obj) / (eps[2]*eps[2]) )
            
            return ret

        cur = correctTrans(cur, config)
        
        eps = [0.5, 0.5, 0.5]
        ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 20, 'eps': eps})

        eps = [0.05, 0.05, 0.05]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 40, 'eps': eps})
        eps = [0.005, 0.005, 0.005]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 40, 'eps': eps})
        eps = [0.0025, 0.0025, 0.0025]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 40, 'eps': eps})

        cur = applyRot(cur, ret.x[0], ret.x[1], ret.x[2])

        #cur = correctTrans(cur, config)
        
        config["angle_noise"] += ret.x
    
    else:
        def f(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            cur_x = applyRot(cur_x, x[3], x[4], x[5])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
            ret = 1-calcGIObjective(real_img, proj, config)
            #print(ret)
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            cur_x = applyRot(cur_x, x[3], x[4], x[5])
            dvec = [cur_x]
            dvec.append(applyTrans(cur_x, eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[2]))
            dvec.append(applyRot(cur_x, eps[3], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[4], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[5]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            h0 = 1-calcGIObjective(real_img, projs[:,0], config)
            #ret = [0,0,0,0,0,0]
            ret = [0,0,0]
            for i in range(1, projs.shape[1]):
                ret[i-1] = (1-calcGIObjective(real_img, projs[:,i], config)-h0) * eps[i-1]
            
            #print(ret)

        def hessf(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyRot(cur_x, eps[0], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]))
            dvec.append(applyRot(cur_x, eps[0]*2, 0, 0))
            dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, 0, eps[1]*2, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]*2))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            h0 = 1-calcGIObjective(real_img, projs[:,0], config)

            objs = np.zeros((9,))
            for i in range(1, projs.shape[1]):
                objs[i-1] = 1-calcGIObjective(real_img, projs[:,i], config)
            
            ret = np.zeros((3,3))
            ret[0,0] = (objs[3] - objs[0] - objs[0] + h0) / (eps[0]*eps[0])
            ret[0,1] = (objs[4] - objs[0] - objs[1] + h0) / (eps[0]*eps[1])
            ret[0,2] = (objs[5] - objs[0] - objs[2] + h0) / (eps[0]*eps[0])
            #ret[1,0] = (objs[4] - objs[1] - objs[0] + h0) / (eps[1]*eps[0])
            ret[1,0] = ret[0,1]
            ret[1,1] = (objs[6] - objs[1] - objs[1] + h0) / (eps[1]*eps[1])
            ret[1,2] = (objs[7] - objs[1] - objs[2] + h0) / (eps[1]*eps[2])
            #ret[2,0] = (objs[5] - objs[2] - objs[0] + h0) / (eps[2]*eps[0])
            ret[2,0] = ret[0,2]
            #ret[2,1] = (objs[7] - objs[2] - objs[1] + h0) / (eps[2]*eps[1])
            ret[2,1] = ret[1,2]
            ret[2,2] = (objs[8] - objs[2] - objs[2] + h0) / (eps[2]*eps[2])
            
            return ret



    cur_x = np.array([0,0,0])
    for its,eps in [(20,0.5), (30,0.1), (30,0.01)]:
        cur = correctTrans(cur, config)
        if config['opti_method'] in ("BFGS","SLSQP", "L-BFGS-B", "TNC"):
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,[eps,eps,eps]),
                                    method=config['opti_method'],
                                    jac=gradf,
                                    options={'maxiter': its, 'eps': [eps,eps,eps]})
        elif config['opti_method'] in ("COBYLA",):
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,[eps,eps,eps]),
                                    method=config['opti_method'],
                                    options={'maxiter': its, 'rhobeg': eps})
        else:
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,[eps,eps,eps]),
                                    method=config['opti_method'],
                                    jac=gradf,
                                    hess=hessf,
                                    options={'maxiter': its})
        cur_x = np.array(ret.x)
        cur = applyRot(cur, cur_x[0], cur_x[1], cur_x[2])

        eps = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
        ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=(cur,eps), method='L-BFGS-B', jac=gradf, options={'maxiter': 20, 'eps': eps})
        eps = [0.05, 0.05, 0.05, 0.005, 0.005, 0.005]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B', jac=gradf, options={'maxiter': 20, 'eps': eps})
        eps = [0.025, 0.025, 0.025, 0.0025, 0.0025, 0.0025]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B', jac=gradf, options={'maxiter': 20, 'eps': eps})
        
        cur = applyTrans(cur, ret.x[0], ret.x[1], ret.x[2])
        cur = applyRot(cur, ret.x[3], ret.x[4], ret.x[5])

        config["trans_noise"] += ret.x[:3]
        config["angle_noise"] += ret.x[3:]

    reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return cur


def bfgs_trans(cur, reg_config, c):
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c==1

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)
    
    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    data_real = config["data_real"]
    points_real = normalize_points(data_real[0], real_img)
    if config["my"]:
        def f(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]

            (p,v), proj = trackFeatures(proj, data_real, config), proj
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            ret = 0
            for axis, mult in [(-1,1),(-2,1),(-3,3)]:
                obj = calcPointsObjective(axis, points, points_real[valid])
                if obj==-1:
                    ret += 50*mult
                else:
                    ret += obj*mult
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyTrans(cur_x, eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, -eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 2*eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 2*-eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, -eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 2*eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 2*-eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[2]))
            dvec.append(applyTrans(cur_x, 0, 0, -eps[2]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*eps[2]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[2]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            ret = [0,0,0]
            for j in range(3):
                i = j*4+1
                def calc_obj(i):
                    proj = projs[:,i]
                    (p,v), proj = trackFeatures(proj, data_real, config), proj
                    points = normalize_points(p, proj)
                    valid = v==1
                    points = points[valid]
                    axis, mult = [(-1,1),(-2,1),(-3,1)][j]
                    obj = calcPointsObjective(axis, points, points_real[valid])*mult
                    if obj==-1:
                        obj = 50
                    return obj
                
                h1 = calc_obj(i)
                h_1 = calc_obj(i+1)
                h2 = calc_obj(i+2)
                h_2 = calc_obj(i+3)
                ret[j] = (-h2+8*h1-8*h_1+h_2)/12

            return ret

        #cur = correctXY(cur, config)
        #cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
        
        eps = [0.5, 0.5, 5]
        ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 50, 'eps': eps})

        eps = [0.25, 0.25, 0.5]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 60, 'eps': eps})
        eps = [0.05, 0.05, 0.25]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 70, 'eps': eps})

        cur = applyTrans(cur, ret.x[0], ret.x[1], ret.x[2])

        #cur = correctXY(cur, config)
        #cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
        
        config["trans_noise"] += ret.x
    
    else:
        
        config["GIoldold"] = None

        def f(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
            ret = calcGIObjective(real_img, proj, config)
            #print(ret)
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyTrans(cur_x, eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, -eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*-eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, -eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*-eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, -eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[2]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            #import SimpleITK as sitk
            #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/jac.nrrd")
            #exit()

            h0 = calcGIObjective(real_img, projs[:,0], config)
            ret = [0,0,0]
            for i in range(1, projs.shape[1]):
            #for j in range(3):
            #    i = j*4+1
                ret[i-1] = (calcGIObjective(real_img, projs[:,i], config)-h0) * 0.5
            #    h1 = (calcGIObjective(real_img, projs[:,i], config))
            #    h_1 = (calcGIObjective(real_img, projs[:,i+1], config))
            #    h2 = (calcGIObjective(real_img, projs[:,i+2], config))
            #    h_2 = (calcGIObjective(real_img, projs[:,i+3], config))
            #    ret[j] = (-h2+8*h1-8*h_1+h_2)/12
            
            #print(ret)

            return ret


        if c==2:
            eps = [1, 1, 5]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.5, 0.5, 1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
            
            eps = [0.1, 0.1, 0.5]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
        elif c==-2:
            eps = [1, 1, 5]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 150, 'eps': eps})

            eps = [0.5, 0.5, 1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 150, 'eps': eps})
        elif c==-3:
            eps = [0.5, 0.5, 2]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
        elif c==-4:
            eps = [0.5, 0.5, 2]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 10, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 10, 'eps': eps})
        elif c==-5:
            eps = [0.5, 0.5, 2]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.25, 0.25, 0.5]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

    cur = applyTrans(cur, ret.x[0], ret.x[1], ret.x[2])

    config["trans_noise"] += ret.x[:3]

    reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return cur

def calc_obj(cur_proj, i, k, config):
    config["data_real"][i] = findInitialFeatures(config["real_img"][i], config)
    config["points_real"][i] = normalize_points(config["data_real"][i][0], config["real_img"][i])

    (p,v) = trackFeatures(cur_proj, config["data_real"][i], config)
    points = normalize_points(p, cur_proj)
    valid = v==1
    points = points[valid]
    axis, mult = [(-1,1),(-2,1),(-3,1)][k]
    obj = calcPointsObjective(axis, points, config["points_real"][i][valid])*mult
    if obj<0:
        obj = 50
    return obj

def t_obj(q,indices,config,proj):
    np.seterr("raise")
    if config["my"]:
        for i in indices:
            q.put(calc_obj(proj[:,i], i, i%3, config))
    else:
        for i in indices:
            q.put(calcGIObjective(config["real_img"][i], proj[:,i], i, None, config))

def t_obj1(q,indices,config,projs):
    np.seterr("raise")
    if config["my"]:
        for j in indices:
            pos = j*4
            for i in range(3):
                h0 = calc_obj(projs[:,pos], j, i, config)
                q.put((j*3+i, (calc_obj(projs[:,pos+i+1], j, i, config)-h0)*0.5))
    else:
        for j in indices:
            pos = j*4
            h0 = calcGIObjective(config["real_img"][j], projs[:,pos], j, None, config)
            for i in range(3):
                q.put((j*3+i, (calcGIObjective(config["real_img"][j], projs[:,pos+i+1], j, None, config)-h0) * 0.5))

def t_obj2(q,indices,config,projs):
    np.seterr("raise")
    if config["my"]:
        for j in indices:
            pos = j*12
            for i in range(3):
                h1 = calc_obj(projs[:,pos+i*4], j, i, config)
                h_1 = calc_obj(projs[:,pos+i*4+1], j, i, config)
                h2 = calc_obj(projs[:,pos+i*4+2], j, i, config)
                h_2 = calc_obj(projs[:,pos+i*4+3], j, i, config)
                q.put((j*3+i, (-h2+8*h1-8*h_1+h_2)/12))
    else:
        for j in indices:
            pos = j*12
            for i in range(3):
                h1 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4], j, None, config))
                h_1 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+1], j, None, config))
                h2 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+2], j, None, config))
                h_2 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+3], j, None, config))
                q.put((j*3+i, (-h2+8*h1-8*h_1+h_2)/12))


def bfgs_trans_all(curs, reg_config, c):
    global gis
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c<=-30

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    def f(x, curs, eps):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x.append(applyTrans(cur, x[pos], x[pos+1], x[pos+2]))
        cur_x = np.array(cur_x)
        proj = Projection_Preprocessing(Ax(cur_x))
        
        q = mp.Queue()

        ts = []
        for u in np.array_split(list(range(len(curs))), 8):
            #t = threading.Thread(target=t_obj, args = (q, u))
            #t.daemon = True
            t = mp.Process(target=t_obj, args = (q,u,filt_conf(config),proj))
            t.start()
            ts.append(t)
        for _ in range(proj.shape[1]):
            ret += q.get()
        
        for t in ts:
            t.join()
        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret/len(curs)

    def f_(x, curs, eps):
        perftime = time.perf_counter() # 185.5 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x.append(applyTrans(cur, x[pos], x[pos+1], x[pos+2]))
        proj = Projection_Preprocessing(Ax(np.array(cur_x)))
        if config["my"]:
            for i in range(len(curs)):
                ret += calc_obj(proj[:,i], i, i%3)
        else:
            for i in range(len(curs)):
                ret += calcGIObjective(real_img[i], proj[:,i], i, cur_x[i], config)
        
        print("obj_", time.perf_counter()-perftime)
        return ret

    def gradf(x, curs, eps):
        perftime = time.perf_counter() # 150 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
            dvec.append(cur_x)
            dvec.append(applyTrans(cur_x, eps[pos], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[pos+1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[pos+2]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        ret = [0,0,0]*len(curs)
        ret_set = np.zeros(len(ret), dtype=bool)
        
        q = mp.Queue()
        
        ts = []
        for u in np.array_split(list(range(real_img.shape[0])), 8):
            #t = threading.Thread(target=t_obj, args = (q, u))
            #t.daemon = True
            t = mp.Process(target=t_obj1, args = (q, u, filt_conf(config), projs))
            t.start()
            ts.append(t)

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res/len(curs)
            ret_set[i] = True
        
        if not ret_set.all():
            print("not all grad elements were set")
        
        for t in ts:
            t.join()
        #print("grad", time.perf_counter()-perftime)
        return ret
    
    def gradf3(x, curs, eps):
        perftime = time.perf_counter() # 150 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
            #dvec.append(cur_x)
            dvec.append(applyTrans(cur_x, eps[pos], 0, 0))
            dvec.append(applyTrans(cur_x, -eps[pos], 0, 0))
            dvec.append(applyTrans(cur_x, 2*eps[pos], 0, 0))
            dvec.append(applyTrans(cur_x, 2*-eps[pos], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[pos+1], 0))
            dvec.append(applyTrans(cur_x, 0, -eps[pos+1], 0))
            dvec.append(applyTrans(cur_x, 0, 2*eps[pos+1], 0))
            dvec.append(applyTrans(cur_x, 0, 2*-eps[pos+1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[pos+2]))
            dvec.append(applyTrans(cur_x, 0, 0, -eps[pos+2]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*eps[pos+2]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[pos+2]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        ret = [0,0,0]*len(curs)
        ret_set = np.zeros(len(ret), dtype=bool)
        
        q = mp.Queue()
        
        ts = []
        for u in np.array_split(list(range(real_img.shape[0])), 8):
            #t = threading.Thread(target=t_obj2, args = (q, u))
            #t.daemon = True
            t = threading.Thread(target=t_obj2, args = (q, u, filt_conf(config), projs))
            t.start()
            ts.append(t)

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res/len(curs)
            ret_set[i] = True
        
        if not ret_set.all():
            print("not all grad elements were set")

        for t in ts:
            t.join()
            
        #print("grad", time.perf_counter()-perftime)
        return ret

    def gradf_(x, curs, eps):
        perftime = time.perf_counter() # 525 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
            dvec.append(cur_x)
            dvec.append(applyTrans(cur_x, eps[pos], 0, 0))
            #dvec.append(applyTrans(cur_x, -eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*-eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[pos+1], 0))
            #dvec.append(applyTrans(cur_x, 0, -eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*-eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[pos+2]))
            #dvec.append(applyTrans(cur_x, 0, 0, -eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[2]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        #import SimpleITK as sitk
        #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/jac.nrrd")
        #exit()

        ret = [0,0,0]*len(curs)
        for j in range(len(curs)):
            pos = j*4
            h0 = calcGIObjective(real_img[j], projs[:,pos], j, dvec[pos], config)
            for i in range(3):
                ret[j*3+i] = (calcGIObjective(real_img[j], projs[:,pos+i+1], j, dvec[pos+i+1], config)-h0) * 0.5
            #    h1 = (calcGIObjective(real_img, projs[:,i], config))
            #    h_1 = (calcGIObjective(real_img, projs[:,i+1], config))
            #    h2 = (calcGIObjective(real_img, projs[:,i+2], config))
            #    h_2 = (calcGIObjective(real_img, projs[:,i+3], config))
            #    ret[j] = (-h2+8*h1-8*h_1+h_2)/12
            
            #print(ret)

        print("grad", time.perf_counter()-perftime)
        return ret


    if config["my"]:
        #if "data_real" not in config or config["data_real"] is None:
        data_real = []
        for img in real_img:
            data_real.append(findInitialFeatures(img, config))
        #config["data_real"] = np.array(real_data)
        config["data_real"] = data_real
        config["points_real"] = [normalize_points(data_real[i][0], real_img[i]) for i in range(len(data_real))]

        if c==-34:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})

            eps = [0.5, 0.5, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
            eps = [0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
        elif c==-37:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
        elif c==-39:
            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})

    else:
        
        config["GIoldold"] = [None]*len(curs)
        config["absp1"] = [None]*len(curs)
        config["p1"] = [None]*len(curs)
        gis = [{} for _ in range(len(curs))]


        if c==2:
            eps = [1, 1, 5]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.5, 0.5, 1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
            
            eps = [0.1, 0.1, 0.5]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
        elif c==-22:
            eps = [1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-23:
            eps = [1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-24:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-25:
            eps = [1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-26:
            eps = [1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-27:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
        elif c==-29:
            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})



        else:
            print("no method selected", c)
    
    res = []
    for i, cur in enumerate(curs):
        res.append(applyTrans(cur, ret.x[i*3+0], ret.x[i*3+1], ret.x[i*3+2]))

    #config["trans_noise"] += ret.x[:3]
    trans_noise += ret.x.reshape(trans_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return np.array(res), (trans_noise, angles_noise)

pdim = 95
sdim = 95
tdim = 3

def simulate_est_data(cur, Ax, config=None):
    if config is None:
        config = dict(default_config)
    
    cur = np.zeros((3, 3), dtype=float)
    cur[1,0] = 1
    cur[2,1] = 1

    primary = np.linspace(0, 360, pdim, False)
    #primary = [0]
    secondary = np.linspace(0, 360, sdim, False)
    #tertiary = [0]
    tertiary = np.linspace(0, 360, tdim, False)
    #tertiary = [0]
    
    bp = 90
    bs = 10
    bt = -90

    pos = []
    projs_data = []
    for t in tertiary:
        print(t)
        curs = []
        for p in primary:
            for s in secondary:
                dcur = applyRot(cur, p+bp, s+bs, t+bt)
                curs.append(dcur)
                pos.append([p+bp, s+bs, t+bt])
        curs = np.array(curs)
        projs = Projection_Preprocessing(Ax(curs))
        for i in range(projs.shape[1]):
            proj = projs[:,i]
            projs_data.append(findInitialFeatures(proj, config))
        
    pos = np.array(pos)
    return pos, projs_data


def est_position(in_cur, Ax, real_imgs, est_data):
    cur = np.array(in_cur)
    config = dict(default_config)

    #perftime = time.perf_counter()
    #print("simulate and find", end=' ')
    #if est_data is None:
        #pos, projs_data = simulate_est_data(cur, Ax, config)
    #else:
    pos, projs_data = est_data
    #print(time.perf_counter()-perftime)
    #perftime = time.perf_counter()
    #print("evaluate", end=' ')
    curs = []
    poss = []
    vmax = 0
    index = (0,0,0)
    cur0 = np.zeros((3, 3), dtype=float)
    cur0[1,0] = 1
    cur0[2,1] = 1
    for real_img in real_imgs:
        data_real = findInitialFeatures(real_img, config)
        points_real = normalize_points(data_real[0], real_img)
        no_valid = []
        indexes = []
        for i in range(0, pdim, 4):
            for j in range(0, sdim, 4):
                for k in range(0, tdim, 1):
                    #proj = projs[:,i]
                    #(p,v) = trackFeatures(proj, data_real, config)
                    idx = k*pdim*sdim+i*sdim+j
                    #config["prefix"] = "debug_imgs\\{:0>3}_{:0>3}_{:0>3}".format(np.round(pos[idx][0]),np.round(pos[idx][1]),np.round(pos[idx][2]))
                    #config["real_img"] = real_img
                    #proj = Projection_Preprocessing(Ax(np.array([applyRot(cur0, pos[idx][0],pos[idx][1],pos[idx][2])])))
                    #(p,v) = matchFeatures(data_real, projs_data[idx], config, proj[:,0])
                    (p,v) = matchFeatures(data_real, projs_data[idx], config={"lowe_ratio": 0.8})
                    valid = np.count_nonzero(v==1)
                    #points_new = normalize_points(p[v], real_img)
                    #valid = calcPointsObjective(-6, points_new, points_real[v])
                    no_valid.append(valid)
                    indexes.append((i,j,k))
                    if vmax==0 or valid > vmax:
                        vmax = valid
                        index = (i,j,k)
        #np.sort(no_valid)[::-1]
        index2 = index
        indexes = np.array(indexes)
        vmax = 0
        indexes2 = set()
        #count = {}
        #print(pos[ np.array([x[0]*sdim+x[1]+x[2]*pdim*sdim for x in indexes[np.argsort(no_valid)[-4:]]])] )
        for p in np.argsort(no_valid)[-5:]:
            index = indexes[p]
            #if (index[0],index[1]) not in count:
            #    count[(index[0],index[1])]=[0,0]
            #count[(index[0],index[1])][0] += 1
            #count[(index[0],index[1])][1] += no_valid[p]
            for i in range(index[0]-4,index[0]+5,1):
                if i < 0 or i >= pdim: continue
                for j in range(index[1]-4,index[1]+5,1):
                    if j < 0 or j >= sdim: continue
                    for k in range(index[2]-4,index[2]+5,1):
                        if k<0 or k>=tdim: continue
                        indexes2.add((i,j,k))
        #index3 = indexes[np.argsort(no_valid)[-1]]
        #count2 = {}
        for (i,j,k) in indexes2:
            (p,v) = matchFeatures(data_real, projs_data[k*sdim*pdim+i*sdim+j], config={"lowe_ratio": 0.7})
            #if (i,j) not in count2:
            #    count2[(i,j)] = [0,0]
            valid = np.count_nonzero(v==1)
            #count2[(i,j)][0] += 1
            #count2[(i,j)][1] += valid
            #points_new = normalize_points(p[v], real_img)
            #valid = calcPointsObjective(-6, points_new, points_real[v])
            #no_valid.append(valid)
            if vmax == 0 or valid > vmax:
                vmax = valid
                index2 = (i,j,k)

        index = index2

        #no_valid = np.array(no_valid)

        #lv = np.argsort(no_valid)[::-1]
        #b = lv[0]
        #b = index[2]*sdim*pdim+index[0]*sdim+index[1]
        #print(time.perf_counter()-perftime)
        #perftime = time.perf_counter()
        #print("rotate", end=' ')

        #bp, bs, bt = pos[b]

        #i=sorted(count2.items(), key=lambda x: x[1][1])[-1]
        #poss.append(np.array([bp, bs, bt]))
        #index4=pos[pdim*sdim+i[0][0]*sdim+i[0][1]]
        index4 = pos[index[2] *sdim*pdim+index[0]*sdim+index[1]]
        bp, bs, bt = index4

        if False:
            #(p,v) = matchFeatures(data_real, projs_data[b], config)
            #print(np.count_nonzero(v), projs_data[b][0].shape, data_real[0].shape)
            config["lowe_ratio"] = 0.75
            (p,v) = matchFeatures(data_real, projs_data[b], config={"lowe_ratio": 0.75})
            #print(np.count_nonzero(v), projs_data[b][0].shape, data_real[0].shape)
            #print(p.shape, v.shape, projs_data[b][0].shape, data_real[0].shape)
            points_new = normalize_points(p[v], real_img)
            #points_real = normalize_points(projs_data[b][0][v], real_img)
            points_r = points_real[v]
            
            new_mid = np.mean(points_new, axis=0)
            real_mid = np.mean(points_r, axis=0)

            points_new = points_new - new_mid
            points_r = points_r - real_mid

            #c = np.linalg.norm(points_new-points_real, axis=-1)
            #a = np.linalg.norm(points_new, axis=-1)
            #b = np.linalg.norm(points_real, axis=-1)
            #print(a.shape, b.shape, points_new.shape)

            #angle = np.arccos((a*a+b*b-c*c) / (2*a*b))*180.0/np.pi
            #val = (points_new[:,0]*points_real[:,0]+points_new[:,1]*points_real[:,1]) / (a*b)
            #angle_cos = np.arccos( val )*180.0/np.pi
            angle = (np.arctan2(points_new[:,0], points_new[:,1])-np.arctan2(points_r[:,0], points_r[:,1])) * 180.0/np.pi
            angle[angle<-180] += 360
            angle[angle>180] -= 360

            #print(np.min(angle), np.mean(angle), np.median(angle), np.max(angle))
            #print(np.min(angle_cos), np.mean(angle_cos), np.median(angle_cos), np.max(angle_cos))
            projs = Projection_Preprocessing(Ax(np.array([applyRot(cur, 0,0,-np.median(angle)), applyRot(cur, 0,0,np.median(angle)) ]))) #, applyRot(cur, 180,0,-np.median(angle)), applyRot(cur, 180,0,np.median(angle))])))
            p,v = trackFeatures(projs[:,0], data_real, config)
            points = normalize_points(p, projs[:,0])
            valid = v==1
            diffn = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))
            p,v = trackFeatures(projs[:,1], data_real, config)
            points = normalize_points(p, projs[:,1])
            valid = v==1
            diffp = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))
            #p,v = trackFeatures(projs[:,0], data_real, config)
            #points = normalize_points(p, projs[:,2])
            #valid = v==1
            #diffnf = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))
            #p,v = trackFeatures(projs[:,1], data_real, config)
            #points = normalize_points(p, projs[:,3])
            #valid = v==1
            #diffpf = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))

            if diffn < diffp:
                bt = bt - np.median(angle)
            else:
                bt = bt + np.median(angle)

        #flip = False
        #if diffn < diffnf and diffn < diffpf and diffn < diffp:
        #    bt = -np.median(angle)
        #if diffp < diffnf and diffp < diffpf and diffp < diffn:
        #    bt = np.median(angle)
        #if diffnf < diffn and diffnf < diffpf and diffnf < diffp:
        #    bt = np.median(angle)
        #    flip = True
        #if diffpf < diffnf and diffpf < diffn and diffpf < diffp:
        #    bt = -np.median(angle)
        #    flip = True

        #print(time.perf_counter()-perftime)
        #if flip:
        #    curs[-1] = applyRot(curs[-1], 180, 0, 0)


        
        poss.append(np.array([bp,bs,bt]))
        #if bp!=index4[0] or bs !=index4[1] or bt!=index4[2]:
        #    print([(list(pos[i[0][0]*sdim+i[0][1]]), i[1]) for i in sorted(count.items(), key=lambda x: x[1][1])], [(list(pos[i[0][0]*sdim+i[0][1]]), i[1]) for i in sorted(count2.items(), key=lambda x: x[1][1])[-3:]])
        #    print(list(pos[index3[2]*sdim*pdim+index3[0]*sdim+index3[1]]), list(pos[index2[2]*sdim*pdim+index2[0]*sdim+index2[1]]), list(index4), [bp, bs, bt])
        
        curs.append(applyRot(in_cur, poss[-1][0], poss[-1][1], poss[-1][2]))
    #plt.figure()
    #plt.title(title)
    #plt.plot(np.arange(len(no_valid)), no_valid)
    return np.array(curs), np.array(poss)


def est_position_ngi(in_cur, Ax, real_imgs, est_data):
    cur = np.array(in_cur)
    config = dict(default_config)

    #perftime = time.perf_counter()
    #print("simulate and find", end=' ')
    #if est_data is None:
        #pos, projs_data = simulate_est_data(cur, Ax, config)
    #else:
    pos, projs_data = est_data
    #print(time.perf_counter()-perftime)
    #perftime = time.perf_counter()
    #print("evaluate", end=' ')
    curs = []
    poss = []
    vmax = 0
    index = (0,0,0)
    cur0 = np.zeros((3, 3), dtype=float)
    cur0[1,0] = 1
    cur0[2,1] = 1
    for real_img in real_imgs:
        data_real = findInitialFeatures(real_img, config)
        points_real = normalize_points(data_real[0], real_img)
        no_valid = []
        indexes = []
        for i in range(0, pdim, 4):
            for j in range(0, sdim, 4):
                for k in range(0, tdim, 1):
                    idx = k*pdim*sdim+i*sdim+j
                    values = np.array([calcGIObjective(real_img, projs[:,idx], 0, None, config) for i in range(projs.shape[1])])
                    (p,v) = matchFeatures(data_real, projs_data[idx], config={"lowe_ratio": 0.8})
                    valid = np.count_nonzero(v==1)
                    #points_new = normalize_points(p[v], real_img)
                    #valid = calcPointsObjective(-6, points_new, points_real[v])
                    no_valid.append(valid)
                    indexes.append((i,j,k))
                    if vmax==0 or valid > vmax:
                        vmax = valid
                        index = (i,j,k)
        #np.sort(no_valid)[::-1]
        index2 = index
        indexes = np.array(indexes)
        vmax = 0
        indexes2 = set()
        #count = {}
        #print(pos[ np.array([x[0]*sdim+x[1]+x[2]*pdim*sdim for x in indexes[np.argsort(no_valid)[-4:]]])] )
        for p in np.argsort(no_valid)[-5:]:
            index = indexes[p]
            #if (index[0],index[1]) not in count:
            #    count[(index[0],index[1])]=[0,0]
            #count[(index[0],index[1])][0] += 1
            #count[(index[0],index[1])][1] += no_valid[p]
            for i in range(index[0]-4,index[0]+5,1):
                if i < 0 or i >= pdim: continue
                for j in range(index[1]-4,index[1]+5,1):
                    if j < 0 or j >= sdim: continue
                    for k in range(index[2]-4,index[2]+5,1):
                        if k<0 or k>=tdim: continue
                        indexes2.add((i,j,k))
        #index3 = indexes[np.argsort(no_valid)[-1]]
        #count2 = {}
        for (i,j,k) in indexes2:
            (p,v) = matchFeatures(data_real, projs_data[k*sdim*pdim+i*sdim+j], config={"lowe_ratio": 0.7})
            #if (i,j) not in count2:
            #    count2[(i,j)] = [0,0]
            valid = np.count_nonzero(v==1)
            #count2[(i,j)][0] += 1
            #count2[(i,j)][1] += valid
            #points_new = normalize_points(p[v], real_img)
            #valid = calcPointsObjective(-6, points_new, points_real[v])
            #no_valid.append(valid)
            if vmax == 0 or valid > vmax:
                vmax = valid
                index2 = (i,j,k)

        index = index2

        #no_valid = np.array(no_valid)

        #lv = np.argsort(no_valid)[::-1]
        #b = lv[0]
        #b = index[2]*sdim*pdim+index[0]*sdim+index[1]
        #print(time.perf_counter()-perftime)
        #perftime = time.perf_counter()
        #print("rotate", end=' ')

        #bp, bs, bt = pos[b]

        #i=sorted(count2.items(), key=lambda x: x[1][1])[-1]
        #poss.append(np.array([bp, bs, bt]))
        #index4=pos[pdim*sdim+i[0][0]*sdim+i[0][1]]
        index4 = pos[index[2] *sdim*pdim+index[0]*sdim+index[1]]
        bp, bs, bt = index4

        if False:
            #(p,v) = matchFeatures(data_real, projs_data[b], config)
            #print(np.count_nonzero(v), projs_data[b][0].shape, data_real[0].shape)
            config["lowe_ratio"] = 0.75
            (p,v) = matchFeatures(data_real, projs_data[b], config={"lowe_ratio": 0.75})
            #print(np.count_nonzero(v), projs_data[b][0].shape, data_real[0].shape)
            #print(p.shape, v.shape, projs_data[b][0].shape, data_real[0].shape)
            points_new = normalize_points(p[v], real_img)
            #points_real = normalize_points(projs_data[b][0][v], real_img)
            points_r = points_real[v]
            
            new_mid = np.mean(points_new, axis=0)
            real_mid = np.mean(points_r, axis=0)

            points_new = points_new - new_mid
            points_r = points_r - real_mid

            #c = np.linalg.norm(points_new-points_real, axis=-1)
            #a = np.linalg.norm(points_new, axis=-1)
            #b = np.linalg.norm(points_real, axis=-1)
            #print(a.shape, b.shape, points_new.shape)

            #angle = np.arccos((a*a+b*b-c*c) / (2*a*b))*180.0/np.pi
            #val = (points_new[:,0]*points_real[:,0]+points_new[:,1]*points_real[:,1]) / (a*b)
            #angle_cos = np.arccos( val )*180.0/np.pi
            angle = (np.arctan2(points_new[:,0], points_new[:,1])-np.arctan2(points_r[:,0], points_r[:,1])) * 180.0/np.pi
            angle[angle<-180] += 360
            angle[angle>180] -= 360

            #print(np.min(angle), np.mean(angle), np.median(angle), np.max(angle))
            #print(np.min(angle_cos), np.mean(angle_cos), np.median(angle_cos), np.max(angle_cos))
            projs = Projection_Preprocessing(Ax(np.array([applyRot(cur, 0,0,-np.median(angle)), applyRot(cur, 0,0,np.median(angle)) ]))) #, applyRot(cur, 180,0,-np.median(angle)), applyRot(cur, 180,0,np.median(angle))])))
            p,v = trackFeatures(projs[:,0], data_real, config)
            points = normalize_points(p, projs[:,0])
            valid = v==1
            diffn = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))
            p,v = trackFeatures(projs[:,1], data_real, config)
            points = normalize_points(p, projs[:,1])
            valid = v==1
            diffp = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))
            #p,v = trackFeatures(projs[:,0], data_real, config)
            #points = normalize_points(p, projs[:,2])
            #valid = v==1
            #diffnf = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))
            #p,v = trackFeatures(projs[:,1], data_real, config)
            #points = normalize_points(p, projs[:,3])
            #valid = v==1
            #diffpf = np.sum(np.abs(np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])))

            if diffn < diffp:
                bt = bt - np.median(angle)
            else:
                bt = bt + np.median(angle)

        #flip = False
        #if diffn < diffnf and diffn < diffpf and diffn < diffp:
        #    bt = -np.median(angle)
        #if diffp < diffnf and diffp < diffpf and diffp < diffn:
        #    bt = np.median(angle)
        #if diffnf < diffn and diffnf < diffpf and diffnf < diffp:
        #    bt = np.median(angle)
        #    flip = True
        #if diffpf < diffnf and diffpf < diffn and diffpf < diffp:
        #    bt = -np.median(angle)
        #    flip = True

        #print(time.perf_counter()-perftime)
        #if flip:
        #    curs[-1] = applyRot(curs[-1], 180, 0, 0)


        
        poss.append(np.array([bp,bs,bt]))
        #if bp!=index4[0] or bs !=index4[1] or bt!=index4[2]:
        #    print([(list(pos[i[0][0]*sdim+i[0][1]]), i[1]) for i in sorted(count.items(), key=lambda x: x[1][1])], [(list(pos[i[0][0]*sdim+i[0][1]]), i[1]) for i in sorted(count2.items(), key=lambda x: x[1][1])[-3:]])
        #    print(list(pos[index3[2]*sdim*pdim+index3[0]*sdim+index3[1]]), list(pos[index2[2]*sdim*pdim+index2[0]*sdim+index2[1]]), list(index4), [bp, bs, bt])
        
        curs.append(applyRot(in_cur, poss[-1][0], poss[-1][1], poss[-1][2]))
    #plt.figure()
    #plt.title(title)
    #plt.plot(np.arange(len(no_valid)), no_valid)
    return np.array(curs), np.array(poss)
