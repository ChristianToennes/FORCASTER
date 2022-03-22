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
from utils import minimize_log as ml
from feature_matching import *
from simple_cal import *
from objectives import *
from skimage.metrics import structural_similarity,normalized_root_mse
import datetime

def calc_obj(cur_proj, i, k, config):
    if config["data_real"][i] is None:
        config["data_real"][i] = findInitialFeatures(config["real_img"][i], config)
        config["points_real"][i] = normalize_points(config["data_real"][i][0], config["real_img"][i])

    (p,v) = trackFeatures(cur_proj, config["data_real"][i], config)
    valid = v==1
    p = p[valid]
    points = normalize_points(p, cur_proj)
    axis, mult = config["comps"][k]
    obj = calcPointsObjective(axis, points, config["points_real"][i][valid])*mult
    if obj<0:
        obj = 50
    return obj

def t_err(q,indices, config, proj):
    for k,i in enumerate(indices):
        q.put((structural_similarity(config["target_sino"][:,i], proj[:,k]), normalized_root_mse(config["target_sino"][:,i], proj[:,k])))
        #q.put((structural_similarity(config["real_img"][i], proj[:,k]), 0))
        #q.put((0, normalized_root_mse(config["real_img"][i], proj[:,k])))

def t_obj(q,indices, config, proj):
    np.seterr("raise")
    if config["my"]:
        for k,i in enumerate(indices):
            q.put(calc_obj(proj[:,k], i, i%3, config))
    else:
        for k,i in enumerate(indices):
            q.put(calcGIObjective(config["real_img"][i], proj[:,k], i, None, config))

def t_grad(q,indices, config, projs):
    np.seterr("raise")
    if config["my"]:
        for k,j in enumerate(indices):
            pos = k*7
            for i in range(6):
                h0 = calc_obj(projs[:,pos], j, i, config)
                q.put((j*6+i, (calc_obj(projs[:,pos+i+1], j, i, config)-h0)*0.5))
    else:
        for k,j in enumerate(indices):
            pos = k*7
            h0 = calcGIObjective(config["real_img"][j], projs[:,pos], j, None, config)
            for i in range(6):
                q.put((j*6+i, (calcGIObjective(config["real_img"][j], projs[:,pos+i+1], j, None, config)-h0) * 0.5))

def t_grad3(q,indices, config, projs):
    np.seterr("raise")
    if config["my"]:
        for k,j in enumerate(indices):
            pos = k*24
            for i in range(6):
                h1 = calc_obj(projs[:,pos+i*4], j, i, config)
                h_1 = calc_obj(projs[:,pos+i*4+1], j, i, config)
                h2 = calc_obj(projs[:,pos+i*4+2], j, i, config)
                h_2 = calc_obj(projs[:,pos+i*4+3], j, i, config)
                q.put((j*6+i, (-h2+8*h1-8*h_1+h_2)/12))
    else:
        for k,j in enumerate(indices):
            pos = k*24
            for i in range(6):
                h1 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4], j, None, config))
                h_1 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+1], j, None, config))
                h2 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+2], j, None, config))
                h_2 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+3], j, None, config))
                q.put((j*6+i, (-h2+8*h1-8*h_1+h_2)/12))

def applyRots(curs, rots):
    res = []
    for i, cur in enumerate(curs):
        cur_rot = applyRot(cur, rots[i*6], rots[i*6+1], rots[i*6+2])
        res.append(applyTrans(cur_rot, rots[i*6+3], rots[i*6+4], rots[i*6+5]))
    return np.array(res)

def bfgs(curs, reg_config, c):
    global gis
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c>-50

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    
    config["GIoldold"] = [None]*len(curs)
    config["absp1"] = [None]*len(curs)
    config["p1"] = [None]*len(curs)
    gis = [{} for _ in range(len(curs))]
    config["comps"] = None


    def e(x, curs, eps, config):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = []
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*6
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(applyTrans(cur_rot, x[pos+3], x[pos+4], x[pos+5]))
        cur_x = np.array(cur_x)
        proj = Ax(cur_x)
        
        q = mp.Queue()
        ts = []
        for u in np.array_split(list(range(len(curs))), 8):
            t = mp.Process(target=t_err, args = (q,u,filt_conf(config),proj[:,u[0]:(u[-1]+1)]))
            t.start()
            ts.append(t)
        for _ in range(proj.shape[1]):
            ret.append(np.array(q.get()))
        
        for t in ts:
            t.join()

        ret = np.array(ret)
        ret = np.mean(ret, axis=0)
        print(datetime.datetime.now(), "error", time.perf_counter()-perftime, ret)
        return ret


    def f(x, curs, eps, config):
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
        ts = []
        for u in np.array_split(list(range(len(curs))), 8):
            t = mp.Process(target=t_obj, args = (q,u,filt_conf(config),proj[:,u[0]:(u[-1]+1)]))
            t.start()
            ts.append(t)
        for _ in range(proj.shape[1]):
            ret += q.get()
        
        for t in ts:
            t.join()

        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret

    def gradf(x, curs, eps, config):
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
        ts = []
        for u in np.array_split(list(range(real_img.shape[0])), 8):
            t = mp.Process(target=t_grad, args = (q, u, filt_conf(config), projs[:,u[0]*7:(u[-1]+1)*7]))
            t.start()
            ts.append(t)

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res
            ret_set[i] = True
        
        if not ret_set.all():
            print("not all grad elements were set")

        for t in ts:
            t.join()

        #print("grad", time.perf_counter()-perftime)
        return ret
    
    def gradf3(x, curs, eps, config):
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
        ts = []
        for u in np.array_split(list(range(real_img.shape[0])), 8):
            t = mp.Process(target=t_grad3, args = (q, u, filt_conf(config), projs[:,u[0]*24:(u[-1]+1)*24]))
            t.start()
            ts.append(t)

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res
            ret_set[i] = True
        
        if not ret_set.all():
            print("not all grad elements were set")

        for t in ts:
            t.join()
        #print("grad", time.perf_counter()-perftime)
        return ret

    if config["my"]:
        #if "data_real" not in config or config["data_real"] is None:
        real_data = []
        for img in real_img:
            real_data.append(findInitialFeatures(img, config))
        config["data_real"] = real_data
        config["points_real"] = [normalize_points(real_data[i][0], real_img[i]) for i in range(len(real_data))]
        config["comps"] = [(-3,1),(-4,1),(-6,1),(-3,1),(-4,1),(-6,1)]
        config_callback = dict(config)
        config_callback["my"] = False
        if c==-43:
            eps = [0.25, 0.25, 0.25, 2, 2, 2] * len(curs)
            name = "-43.err bfgs full my reduced noise 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config_callback), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS', 
                                          jac=gradf3,  callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                          options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.025, 0.025, 0.025, 1, 1, 1] * len(curs)
            name = "-43.err bfgs full my reduced noise 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-44:
            eps = [0.25, 0.25, 0.25, 2, 2, 2] * len(curs)
            name = "-44.err bfgs full my 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config_callback), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            

            #eps = [0.025, 0.025, 0.025, 1, 1, 1] * len(curs)
            #name = "-44.err bfgs full my 2"
            #ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
            #                            options={'maxiter': 50, 'eps': eps, 'disp': True}))
            
            #eps = [0.01, 0.01, 0.01, 0.5, 0.5, 0.5] * len(curs)
            #name = "-44.err bfgs full my 3"
            #ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
            #                            options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-45:
            eps = [0.25, 0.25, 0.25, 2, 2, 2] * len(curs)
            name = "-45.ngi bfgs full my 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config_callback))
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            
            eps = [0.05, 0.05, 0.05, 1, 1, 1] * len(curs)
            name = "-45.ngi bfgs full my 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01, 0.01, 0.01, 1, 1, 1] * len(curs)
            name = "-45.ngi bfgs full my 3"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-46:
            #config["comps"] = [(0,1),(1,1),(12,1),(-1,1),(-2,1),(-3,1)]
            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
        elif c==-47:
            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

    else:
        

        if c==-52:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-53:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-54:
            eps = [0.25, 0.25, 0.25, 1, 1, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.05, 0.05, 0.05, 0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
        elif c==-55:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-56:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5, 0.5, 0.5, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-57:
            eps = [0.25,0.25,0.25,3, 3, 3] * len(curs)
            name = "-57.err bfgs full ngi reduced noise 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.05,0.05,0.05,2, 2, 2] * len(curs)
            name = "-57.err bfgs full ngi reduced noise 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,callback= utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01,0.01,0.01, 1, 1, 1] * len(curs)
            name = "-57.err bfgs full ngi reduced noise 3"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-58:
            eps = [0.25, 0.25, 0.25,3, 3, 3] * len(curs)
            name = "-58.err bfgs full ngi 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.05,0.05,0.05,2, 2, 2] * len(curs)
            name = "-58.err bfgs full ngi 2"
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01,0.01,0.01, 1, 1, 1] * len(curs)
            name = "-58.err bfgs full ngi 3"
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))



        else:
            print("no method selected", c)
    
    res = []
    for i, cur in enumerate(curs):
        cur_rot = applyRot(cur, ret.x[i*6], ret.x[i*6+1], ret.x[i*6+2])
        res.append(applyTrans(cur_rot, ret.x[i*6+3], ret.x[i*6+4], ret.x[i*6+5]))

    #config["trans_noise"] += ret.x[:3]
    #trans_noise += np.array(ret.x[3:]).reshape(trans_n  oise.shape)
    #angles_noise += np.array(ret.x[:3]).reshape(angles_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return np.array(res), (trans_noise, angles_noise)


def t_calc_obj(q_in, q_out, config):
    while True:
        action, data = q_in.get()
        if action == "exit":
            return
        i, j, k, proj = data
        if action == "my":
            ret = calc_obj(proj, i, k, config)
        elif action == "ngi":
            ret = calcGIObjective(config["real_img"][i], proj, i, None, config)
        elif action == "error":
            ret = (structural_similarity(config["target_sino"][:,i], proj), normalized_root_mse(config["target_sino"][:,i], proj))
        else:
            ret = 0
        q_out.put((i, j, k, ret))


def bfgs_single(curs, reg_config, c):
    global gis
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c>-50

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)
    
    config["GIoldold"] = [None]*len(curs)
    config["absp1"] = [None]*len(curs)
    config["p1"] = [None]*len(curs)
    gis = [{} for _ in range(len(curs))]
    config["comps"] = None

    if config["my"]:
        #if "data_real" not in config or config["data_real"] is None:
        real_data = []
        for img in real_img:
            real_data.append(findInitialFeatures(img, config))
        config["data_real"] = real_data
        config["points_real"] = [normalize_points(real_data[i][0], real_img[i]) for i in range(len(real_data))]
        config["comps"] = [(-3,1),(-4,1),(-6,1),(-3,1),(-4,1),(-6,1)]
        config_callback = dict(config)
        config_callback["my"] = False

    q_in = mp.Queue()
    q_out = mp.Queue()

    if config["my"]:
        method = "my"
    else:
        method = "ngi"

    ts = []
    for _ in range(mp.cpu_count()):
        t = mp.Process(target=t_calc_obj, args=(q_in, q_out, filt_conf(config)))
        t.start()
        ts.append(t)

    def e(x, curs, eps, config):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = []
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(cur_rot)
        cur_x = np.array(cur_x)
        projs = Projection_Preprocessing(Ax(cur_x))
        
        for i in range(projs.shape[1]):
            q_in.put(("error",(i,i,i,projs[:,i])))

        for _ in range(projs.shape[1]):
            _, _, _, r = q_out.get()
            ret.append(r)
        
        ret = np.array(ret)
        ret = np.mean(ret, axis=0)
        print(datetime.datetime.now(), time.perf_counter()-perftime, "error", ret)
        return ret

    def f(x, curs, eps, config):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*6
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(applyTrans(cur_rot, x[pos+3], x[pos+4], x[pos+5]))
        cur_x = np.array(cur_x)
        projs = Projection_Preprocessing(Ax(cur_x))
        
        for i in range(projs.shape[1]):
            q_in.put((method,(i,i,2,projs[:,i])))

        for _ in range(projs.shape[1]):
            _, _, _, r = q_out.get()
            ret += r
        
        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret

    def gradf(x, curs, eps, config):
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

        objs = np.zeros(len(curs)*12, dtype=float)
        
        for i in range(len(curs)):
            for k in range(6):
                j = i*12+k*2
                q_in.put((method,(i,j,k,projs[:,i*7])))
                j = i*12+k*2+1
                q_in.put((method,(i,j,k,projs[:,i*7+k])))

        for _ in range(projs.shape[1]):
            i, j, k, r = q_out.get()
            objs[j] = r
            
        ret = np.zeros(len(curs)*6, dtype=float)
        for i in range(len(ret)):
            h0 = objs[i*2]
            h1 = objs[i*2 + 1]
            ret[i] = 0.5*(h1-h0)
        #print("grad", time.perf_counter()-perftime)
        return ret
    
    def gradf3(x, curs, eps, config):
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

        objs = np.zeros(len(dvec), dtype=float)
        
        for j in range(projs.shape[1]):
            i = j // 12
            k = j % 12
            if k <= 3:
                k = 0
            elif k <= 7:
                k = 1
            elif k <= 11:
                k = 2
            else:
                k = 2
            q_in.put((method,(i,j,k,projs[:,j])))

        for _ in range(projs.shape[1]):
            i, j, k, r = q_out.get()
            objs[j] = r
            
        ret = np.zeros(len(curs)*6, dtype=float)
        for i in range(len(ret)):
            h1 = objs[i*4]
            h_1 = objs[i*4 + 1]
            h2 = objs[i*4 + 2]
            h_2 = objs[i*4 + 3]
            ret[i] = (-h2+8*h1-8*h_1+h_2)/12
        #print("grad", time.perf_counter()-perftime)
        return ret

    if config["my"]:
        if c==-43:
            eps = [0.25, 0.25, 0.25, 2, 2, 2] * len(curs)
            name = "-43.err bfgs full my reduced noise 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config_callback), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS', 
                                          jac=gradf3,  callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                          options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.025, 0.025, 0.025, 1, 1, 1] * len(curs)
            name = "-43.err bfgs full my reduced noise 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-44:
            eps = [0.25, 0.25, 0.25, 2, 2, 2] * len(curs)
            name = "-44.err bfgs full my 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config_callback), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            

            eps = [0.025, 0.025, 0.025, 1, 1, 1] * len(curs)
            name = "-44.err bfgs full my 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            
            eps = [0.01, 0.01, 0.01, 0.5, 0.5, 0.5] * len(curs)
            name = "-44.err bfgs full my 3"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-45:
            eps = [0.25, 0.25, 0.25, 2, 2, 2] * len(curs)
            name = "-45.ngi bfgs full my 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config_callback))
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            
            eps = [0.05, 0.05, 0.05, 1, 1, 1] * len(curs)
            name = "-45.ngi bfgs full my 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01, 0.01, 0.01, 1, 1, 1] * len(curs)
            name = "-45.ngi bfgs full my 3"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

    else:

        if c==-57:
            eps = [0.25,0.25,0.25,3, 3, 3] * len(curs)
            name = "-57.err bfgs full ngi reduced noise 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.05,0.05,0.05,2, 2, 2] * len(curs)
            name = "-57.err bfgs full ngi reduced noise 2"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,callback= utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01,0.01,0.01, 1, 1, 1] * len(curs)
            name = "-57.err bfgs full ngi reduced noise 3"
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-58:
            eps = [0.25, 0.25, 0.25,3, 3, 3] * len(curs)
            name = "-58.err bfgs full ngi 1"
            callback = utils.minimize_callback(name, e, (curs,eps,config), True)
            callback(np.array([0,0,0,0,0,0] * len(curs)))
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.05,0.05,0.05,2, 2, 2] * len(curs)
            name = "-58.err bfgs full ngi 2"
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01,0.01,0.01, 1, 1, 1] * len(curs)
            name = "-58.err bfgs full ngi 3"
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))



        else:
            print("no method selected", c)
    
    res = []
    for i, cur in enumerate(curs):
        cur_rot = applyRot(cur, ret.x[i*6], ret.x[i*6+1], ret.x[i*6+2])
        res.append(applyTrans(cur_rot, ret.x[i*6+3], ret.x[i*6+4], ret.x[i*6+5]))

    
    for t in ts:
        q_in.put(("exit", None))
    #config["trans_noise"] += ret.x[:3]
    #trans_noise += np.array(ret.x[3:]).reshape(trans_n  oise.shape)
    #angles_noise += np.array(ret.x[:3]).reshape(angles_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return np.array(res), (trans_noise, angles_noise)

