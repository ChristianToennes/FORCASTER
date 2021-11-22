import numpy as np
import utils
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import sys
import itertools
import time
import multiprocessing as mp
import queue
from utils import bcolors, applyRot, applyTrans, default_config, filt_conf
from feature_matching import *
from simple_cal import *
from objectives import *

def calc_obj(cur_proj, i, k, config):
    (p,v) = trackFeatures(cur_proj, config["data_real"][i], config)
    valid = v==1
    p = p[valid]
    points = normalize_points(p, cur_proj)
    if "comps" in config:
        axis, mult = config["comps"][k]
    else:
        axis, mult = [(-5,1),(-5,1),(-5,1),(-5,1),(-5,1),(-5,1)][k]
    obj = calcPointsObjective(axis, points, config["points_real"][i][valid])*mult
    if obj<0:
        obj = 50
    return obj

def t_obj(q,indices):
    np.seterr("raise")
    if config["my"]:
        for i in indices:
            q.put(calc_obj(proj[:,i], i, i%6))
    else:
        for i in indices:
            q.put(calcGIObjective(real_img[i], proj[:,i], i, cur_x[i], config))

def t_grad(q,indices):
    np.seterr("raise")
    if config["my"]:
        for j in indices:
            pos = j*7
            for i in range(6):
                h0 = calc_obj(projs[:,pos], j, i)
                q.put((j*6+i, (calc_obj(projs[:,pos+i+1], j, i)-h0)*0.5))
    else:
        for j in indices:
            pos = j*7
            h0 = calcGIObjective(real_img[j], projs[:,pos], j, dvec[pos], config)
            for i in range(6):
                q.put((j*6+i, (calcGIObjective(real_img[j], projs[:,pos+i+1], j, dvec[pos+i+1], config)-h0) * 0.5))

def t_grad3(q,indices):
    np.seterr("raise")
    if config["my"]:
        for j in indices:
            pos = j*24
            for i in range(6):
                h1 = calc_obj(projs[:,pos+i*4], j, i)
                h_1 = calc_obj(projs[:,pos+i*4+1], j, i)
                h2 = calc_obj(projs[:,pos+i*4+2], j, i)
                h_2 = calc_obj(projs[:,pos+i*4+3], j, i)
                q.put((j*6+i, (-h2+8*h1-8*h_1+h_2)/12))
    else:
        for j in indices:
            pos = j*24
            for i in range(6):
                h1 = (calcGIObjective(real_img[j], projs[:,pos+i*4], j, dvec[pos+i], config))
                h_1 = (calcGIObjective(real_img[j], projs[:,pos+i*4+1], j, dvec[pos+i+1], config))
                h2 = (calcGIObjective(real_img[j], projs[:,pos+i*4+2], j, dvec[pos+i+2], config))
                h_2 = (calcGIObjective(real_img[j], projs[:,pos+i*4+3], j, dvec[pos+i+3], config))
                q.put((j*6+i, (-h2+8*h1-8*h_1+h_2)/12))


def bfgs(curs, reg_config, c):
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
            pos = i*6
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(applyTrans(cur_rot, x[pos+3], x[pos+4], x[pos+5]))
        cur_x = np.array(cur_x)
        proj = Projection_Preprocessing(Ax(cur_x))
        
        q = mp.Queue()
        
        for u in np.array_split(list(range(len(curs))), 8):
            t = mp.Process(target=t_grad, args = (q,u,filt_conf(config),proj))
            t.start()
            ts.append(t)
        for _ in range(proj.shape[1]):
            ret += q.get()
        
        for t in ts:
            t.join()

        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret/len(curs)

    def gradf(x, curs, eps):
        perftime = time.perf_counter() # 150 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*6
            cur_x = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x = applyTrans(cur_x, x[pos+3], x[pos+4], x[pos+5])
            dvec.append(cur_x)
            dvec.append(applyRot(cur_x, eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[pos+2]))
            dvec.append(applyTrans(cur_x, eps[pos+3], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[pos+4], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[pos+5]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        ret = [0,0,0,0,0,0]*len(curs)
        ret_set = np.zeros(len(ret), dtype=bool)
        
        q = mp.Queue()
        
        for u in np.array_split(list(range(real_img.shape[0])), 8):
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
            pos = i*6
            cur_x = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x = applyTrans(cur_x, x[pos+3], x[pos+4], x[pos+5])
            dvec.append(applyRot(cur_x, eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, -eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, 2*eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, 2*-eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, -eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, 2*eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, 2*-eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[pos+2]))
            dvec.append(applyRot(cur_x, 0, 0, -eps[pos+2]))
            dvec.append(applyRot(cur_x, 0, 0, 2*eps[pos+2]))
            dvec.append(applyRot(cur_x, 0, 0, 2*-eps[pos+2]))
            dvec.append(applyTrans(cur_x, eps[pos+3], 0, 0))
            dvec.append(applyTrans(cur_x, -eps[pos+3], 0, 0))
            dvec.append(applyTrans(cur_x, 2*eps[pos+3], 0, 0))
            dvec.append(applyTrans(cur_x, 2*-eps[pos+3], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[pos+4], 0))
            dvec.append(applyTrans(cur_x, 0, -eps[pos+4], 0))
            dvec.append(applyTrans(cur_x, 0, 2*eps[pos+4], 0))
            dvec.append(applyTrans(cur_x, 0, 2*-eps[pos+4], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[pos+5]))
            dvec.append(applyTrans(cur_x, 0, 0, -eps[pos+5]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*eps[pos+5]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[pos+5]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        ret = [0,0,0,0,0,0]*len(curs)
        ret_set = np.zeros(len(ret), dtype=bool)
        
        q = mp.Queue()
        
        for u in np.array_split(list(range(real_img.shape[0])), 8):
            t = threading.Thread(target=t_obj, args = (q, u))
            t.daemon = True
            t.start()

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res/len(curs)
            ret_set[i] = True
        
        if not ret_set.all():
            print("not all grad elements were set")

        #print("grad", time.perf_counter()-perftime)
        return ret

    if config["my"]:
        #if "data_real" not in config or config["data_real"] is None:
        real_data = []
        for img in real_img:
            real_data.append(findInitialFeatures(img, config))
        config["data_real"] = np.array(real_data)
        data_real = config["data_real"]
        config["points_real"] = [normalize_points(data_real[i,:,0], real_img[i]) for i in range(data_real.shape[0])]

        if c==-34:
            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
        elif c==-35:
            #config["comps"] = [(10,1),(11,1),(2,1),(-1,1),(-2,1),(-3,1)]
            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
        elif c==-36:
            #config["comps"] = [(0,1),(1,1),(12,1),(-1,1),(-2,1),(-3,1)]
            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
        elif c==-37:
            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

    else:
        
        config["GIoldold"] = [None]*len(curs)
        config["absp1"] = [None]*len(curs)
        config["p1"] = [None]*len(curs)
        gis = [{} for _ in range(len(curs))]

        if c==-22:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-23:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-24:
            eps = [1, 1, 1, 1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01, 0.01, 0.01, 0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-25:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-26:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-27:
            eps = [1,1,1,1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25,0.25,0.25,0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01,0.01,0.01,0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})


        else:
            print("no method selected", c)
    
    res = []
    for i, cur in enumerate(curs):
        cur_rot = applyRot(cur, ret.x[i*6], ret.x[i*6+1], ret.x[i*6+2])
        res.append(applyTrans(cur_rot, ret.x[i*6+3], ret.x[i*6+4], ret.x[i*6+5]))

    #config["trans_noise"] += ret.x[:3]
    trans_noise += ret.x[3:].reshape(trans_noise.shape)
    angles_noise += ret.x[:3].reshape(angles_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return np.array(res), (trans_noise, angles_noise)
