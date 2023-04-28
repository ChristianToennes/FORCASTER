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
import cma
from types import SimpleNamespace

def calc_obj(cur_proj, k, config):
    if config["data_real"] is None:
        config["data_real"] = findInitialFeatures(config["real_img"], config)
        config["points_real"] = normalize_points(config["data_real"][0], config["real_img"])

    (p,v) = trackFeatures(cur_proj, config["data_real"], config)
    valid = v==1
    p = p[valid]
    points = normalize_points(p, cur_proj)
    axis, mult = config["comps"][k]
    obj = calcPointsObjective(axis, points, config["points_real"][valid])*mult
    if obj<0:
        obj = 50
    return obj

def bfgs(cur, reg_config, c):
    global gis
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c>-50
    
    config["my"] = c in [-43,-44,-72]

    real_img = config["real_img"]
    Ax = config["Ax"]

    config["GIoldold"] = [None]
    config["absp1"] = [None]
    config["p1"] = [None]
    gis = [{}]
    config["comps"] = None
    log_queue = config["log_queue"]
    index = str(config["name"])


    def e(x, cur, eps, config):
        #perftime = time.perf_counter() # 100 s / 50 s
        ret = []
        cur_x = []
        cur_rot = applyRot(cur, x[0], x[0+1], x[0+2])
        cur_x.append(applyTrans(cur_rot, x[0+3], x[0+4], x[0+5]))
        cur_x = np.array(cur_x)
        proj = Projection_Preprocessing(Ax(cur_x))
        print(config["real_img"].shape, proj[:,0].shape)
        ret.append( (structural_similarity(config["real_img"], proj[:,0]), normalized_root_mse(config["real_img"], proj[:,0])) )
        #ret.append( (0, normalized_root_mse(config["real_img"], proj[:,0])) )

        return ret

    def f(x, cur, eps, config):
        #perftime = time.perf_counter() # 100 s / 50 s
        ret = 0
        cur_x = []
        cur_rot = applyRot(cur, x[0], x[0+1], x[0+2])
        cur_x.append(applyTrans(cur_rot, x[0+3], x[0+4], x[0+5]))
        cur_x = np.array(cur_x)
        proj = Projection_Preprocessing(Ax(cur_x))
        
        if config["my"]:
            ret = calc_obj(proj[:,0], 2, config)
        else:
            ret = calcGIObjective(config["real_img"], proj[:,0], 0, None, config)

        return ret

    def gradf(x, cur, eps, config):
        #perftime = time.perf_counter() # 150 s
        dvec = []
        pos = 0
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

        ret = [0,0,0,0,0,0]
        
        if config["my"]:
            for i in range(6):
                h0 = calc_obj(projs[:,0], i, config)
                ret[i] = (calc_obj(projs[:,i+1], i, config)-h0)*0.5
        else:
            h0 = calcGIObjective(config["real_img"], projs[:,0], 0, None, config)
            for i in range(6):
                ret[i] = (calcGIObjective(config["real_img"], projs[:,i+1], 0, None, config)-h0) * 0.5

        #print("grad", time.perf_counter()-perftime)
        return ret
    
    def gradf3(x, cur, eps, config):
        #perftime = time.perf_counter() # 150 s
        dvec = []
        pos = 0
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

        ret = [0,0,0,0,0,0]

        if config["my"]:
            for i in range(6):
                h1 = calc_obj(projs[:,i*4], 2, config)
                h_1 = calc_obj(projs[:,i*4+1], 2, config)
                h2 = calc_obj(projs[:,i*4+2], 2, config)
                h_2 = calc_obj(projs[:,i*4+3], 2, config)
                ret[i] =  (-h2+8*h1-8*h_1+h_2)/12
        else:
            for i in range(6):
                h1 = (calcGIObjective(config["real_img"], projs[:,i*4], 0, None, config))
                h_1 = (calcGIObjective(config["real_img"], projs[:,i*4+1], 0, None, config))
                h2 = (calcGIObjective(config["real_img"], projs[:,i*4+2], 0, None, config))
                h_2 = (calcGIObjective(config["real_img"], projs[:,i*4+3], 0, None, config))
                ret[i] = (-h2+8*h1-8*h_1+h_2)/12

        #print("grad", time.perf_counter()-perftime)
        return ret

    ret = object()
    if config["my"]:
        #if "data_real" not in config or config["data_real"] is None:
        real_data = findInitialFeatures(real_img, config)
        config["data_real"] = real_data
        config["points_real"] = normalize_points(real_data[0], real_img)
        config["comps"] = [(-3,1),(-4,1),(-6,1),(-3,1),(-4,1),(-6,1)]
        config_callback = dict(config)
        config_callback["my"] = False
        
        if c==-43:
            eps = [0.25, 0.25, 0.25, 2, 2, 2]
            name = "-43.err " + index + " bfgs full my reduced noise 1"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            callback(np.array([0,0,0,0,0,0]))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=(cur,eps,config), method='BFGS', 
                                          jac=gradf3,  callback=callback,
                                          options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.025, 0.025, 0.025, 1, 1, 1]
            name = "-43.err " + index + " bfgs full my reduced noise 2"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-44:
            eps = [0.25, 0.25, 0.25, 2, 2, 2]
            name = "-44.err " + index + " bfgs full my 1"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            callback(np.array([0,0,0,0,0,0]))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            

            eps = [0.025, 0.025, 0.025, 1, 1, 1]
            name = "-44.err " + index + " bfgs full my 2"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
            
            eps = [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
            name = "-44.err " + index + " bfgs full my 3"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        
        elif c==-72:
            eps = [0.1, 0.1, 0.1, 1, 1, 1]
            #fit = lambda xs: np.array([f(x, cur, eps, config) for x in xs])
            #opt = cma.CMA(initial_solution=np.array([0,0,0,0,0,0]), initial_step_size=1.0, fitness_function=fit, termination_no_effect=0.001)
            #ret, score = opt.search()
            x, score = cma.fmin2(f, np.array([0,0,0,0,0,0]), 1, args=(cur,eps,config), gradf=gradf3, options={'ftarget': 0.001, 'maxiter': 200})
            ret = SimpleNamespace(x=x)
        
        else:
            print("no method selected", c)
       
    else:
        
        if c==-57:
            eps = [0.25,0.25,0.25,3, 3, 3]
            name = "-57.err " + index + " bfgs full ngi reduced noise 1"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            callback(np.array([0,0,0,0,0,0]))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.05,0.05,0.05,2, 2, 2]
            name = "-57.err " + index + " bfgs full ngi reduced noise 2"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3,callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01,0.01,0.01, 1, 1, 1]
            name = "-57.err " + index + " bfgs full ngi reduced noise 3"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret = ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))
        elif c==-58:
            eps = [0.1, 0.1, 0.1,3, 3, 3]
            name = "-58.err " + index + " bfgs full ngi 1"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            callback(np.array([0,0,0,0,0,0]))
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.01,0.01,0.01,2, 2, 2]
            name = "-58.err " + index + " bfgs full ngi 2"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

            eps = [0.001,0.001,0.001, 1, 1, 1]
            name = "-58.err " + index + " bfgs full ngi 3"
            callback = lambda x: log_queue.put((name, e(x, cur, eps, config)))
            ret =  ml(name, time.perf_counter(), scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps,config), method='BFGS',
                                        jac=gradf3, callback=callback,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True}))

        elif c==-70:
            eps = [0.1, 0.1, 0.1, 1, 1, 1]
            cur = np.zeros_like(cur)
            #fit = lambda xs: np.array([f(x, cur, eps, config) for x in xs])
            #opt = cma.CMA(initial_solution=np.array([0,0,0,0,0,0]), initial_step_size=1.0, fitness_function=fit, termination_no_effect=0.001)
            #ret, score = opt.search()
            x, score = cma.fmin2(f, np.array([0,0,0,0,0,0]), 1, args=(cur,eps,config), gradf=gradf3, options={'ftarget': 0.1, 'maxiter': 10})
            eps = [0.01, 0.01, 0.01, 1, 1, 1]
            x, score = cma.fmin2(f, np.array(x), 0.1, args=(cur,eps,config), gradf=gradf3, options={'ftarget': 0.001, 'maxiter': 10})
            ret = SimpleNamespace(x=x)
        elif c==-71:
            eps = [0.1, 0.1, 0.1, 1, 1, 1]
            #fit = lambda xs: np.array([f(x, cur, eps, config) for x in xs])
            #opt = cma.CMA(initial_solution=np.array([0,0,0,0,0,0]), initial_step_size=1.0, fitness_function=fit, termination_no_effect=0.001)
            #ret, score = opt.search()
            x, score = cma.fmin2(f, np.array([0,0,0,0,0,0]), 1, args=(cur,eps,config), gradf=gradf3, options={'ftarget': 0.1, 'maxiter': 10})
            eps = [0.01, 0.01, 0.01, 1, 1, 1]
            x, score = cma.fmin2(f, np.array(x), 0.1, args=(cur,eps,config), gradf=gradf3, options={'ftarget': 0.001, 'maxiter': 10})
            ret = SimpleNamespace(x=x)

        else:
            print("no method selected", c)
    
    cur_rot = applyRot(cur, ret.x[0], ret.x[1], ret.x[2])
    res = applyTrans(cur_rot, ret.x[3], ret.x[4], ret.x[5])

    return res
        
