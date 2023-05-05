import numpy as np
#import cv2
import matplotlib.pyplot as plt
import warnings
import time
from utils import applyRot, default_config
from utils import minimize_log as ml
from feature_matching import *
from simple_cal import *
from objectives import *
from skimage.metrics import structural_similarity,normalized_root_mse
from est_position import est_position

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

    real_img = config["real_img"]
    Ax = config["Ax"]

    #print("rough")
    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)
        
    est_data = config["est_data"]

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
    
    elif c==72:
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
