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
from utils import minimize_log as ml
from utils import bcolors, applyTrans, default_config, filt_conf
from feature_matching import *
from simple_cal import *
from objectives import *

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
            pos = k*4
            for i in range(3):
                h0 = calc_obj(projs[:,pos], j, i, config)
                q.put((j*3+i, (calc_obj(projs[:,pos+i+1], j, i, config)-h0)*0.5))
    else:
        for k,j in enumerate(indices):
            pos = k*4
            h0 = calcGIObjective(config["real_img"][j], projs[:,pos], j, None, config)
            for i in range(3):
                q.put((j*3+i, (calcGIObjective(config["real_img"][j], projs[:,pos+i+1], j, None, config)-h0) * 0.5))

def t_grad3(q,indices, config, projs):
    np.seterr("raise")
    if config["my"]:
        for k,j in enumerate(indices):
            pos = k*12
            for i in range(3):
                h1 = calc_obj(projs[:,pos+i*4], j, i, config)
                h_1 = calc_obj(projs[:,pos+i*4+1], j, i, config)
                h2 = calc_obj(projs[:,pos+i*4+2], j, i, config)
                h_2 = calc_obj(projs[:,pos+i*4+3], j, i, config)
                q.put((j*3+i, (-h2+8*h1-8*h_1+h_2)/12))
    else:
        for k,j in enumerate(indices):
            pos = k*12
            for i in range(3):
                h1 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4], j, None, config))
                h_1 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+1], j, None, config))
                h2 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+2], j, None, config))
                h_2 = (calcGIObjective(config["real_img"][j], projs[:,pos+i*4+3], j, None, config))
                q.put((j*3+i, (-h2+8*h1-8*h_1+h_2)/12))

def applyTranss(curs, transs):
    res = []
    for i, cur in enumerate(curs):
        cur_trans = applyTrans(cur, transs[i*3], transs[i*3+1], transs[i*3+2])
        res.append(cur_trans)
    return np.array(res)

def bfgs(curs, reg_config, c):
    print("bfgs trans all", c)
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c<=-70

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    def f(x, curs, eps, config):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_trans = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(cur_trans)
        cur_x = np.array(cur_x)
        proj = Projection_Preprocessing(Ax(cur_x))
        
        q = mp.Queue()
        
        ts = []
        for u in np.array_split(list(range(len(curs))), 8):
            t = mp.Process(target=t_obj, args = (q,u,filt_conf(config),proj[:,u[0]:(u[-1]+1)]))
            t.start()
            ts.append(t)
        for _ in range(proj.shape[1]):
            r = q.get()
            ret += r
        
        for t in ts:
            t.join()

        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret#/len(curs)

    def gradf(x, curs, eps, config):
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
            t = mp.Process(target=t_grad, args = (q, u, filt_conf(config), projs[:,u[0]*4:(u[-1]+1)*4]))
            t.start()
            ts.append(t)

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res#/len(curs)
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
            pos = i*3
            cur_x = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
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
            t = mp.Process(target=t_grad3, args = (q, u, filt_conf(config), projs[:,u[0]*12:(u[-1]+1)*12]))
            t.start()
            ts.append(t)

        for _ in range(len(ret)):
            i, res = q.get()
            ret[i] = res#/len(curs)
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
        config["data_real"] = np.array(real_data)
        data_real = config["data_real"]
        print(data_real.shape, real_img.shape)
        config["points_real"] = [normalize_points(data_real[i,0], real_img[i]) for i in range(data_real.shape[0])]

        config["comps"] = [(-3,1),(-4,1),(-6,1)]
        ##cur, noise = cal_bfgs_rot.bfgs(params, config, c)

        if c == -70:
            config["comps"] = [(-1,1),(-2,1),(-8,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            eps = [0.1, 0.1, 0.1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            eps = [0.025, 0.025, 0.025] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c == -71:
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c == -72:
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [2, 2, 2] * len(curs)
            starttime = time.perf_counter()
            name = "-72 bfgs trans my reduced noise 1"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            ml(name, starttime, ret)
            starttime = time.perf_counter()
            curs = applyTranss(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [1, 1, 2] * len(curs)
            name = "-72 bfgs trans my reduced noise 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            ml(name, starttime, ret)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c == -73:
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            starttime = time.perf_counter
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [2, 2, 2] * len(curs)
            name = "-73 bfgs trans my 1"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            ml(name, starttime, ret)
            starttime = time.perf_counter()
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [1, 1, 2] * len(curs)
            name = "-73 bfgs trans my 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            ml(name, starttime, ret)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c==-74:
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #eps = [0.1, 0.1, 0.1] * len(curs)
            #ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
            #                            jac=gradf3, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
            #                            options={'maxiter': 100, 'eps': eps, 'disp': True})
            #curs = applyTranss(curs, ret.x)
            eps = [0.025, 0.025, 0.025] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c==-75:
            config["comps"] = [(-1,1),(-2,1),(-8,1)]
            ##cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            #eps = [0.5, 0.5, 0.5] * len(curs)
            #ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf,
            #                            options={'maxiter': 100, 'eps': eps, 'disp': True})
            #curs = applyTranss(curs, ret.x)
            ##cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            #eps = [0.01, 0.01, 0.01] * len(curs)
            #ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf,
            #                            options={'maxiter': 100, 'eps': eps, 'disp': True})
            #curs = applyTranss(curs, ret.x)
            ##cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c==-76:
            config["comps"] = [(-1,1),(-2,1),(-8,1)]
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            eps = [0.1, 0.1, 0.1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c==-77:
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        else:
            print("no method selected", c)
    else:
        
        config["GIoldold"] = [None]*len(curs)
        config["absp1"] = [None]*len(curs)
        config["p1"] = [None]*len(curs)
        gis = [{} for _ in range(len(curs))]
        config["comps"] = None

        if c==-62:
            eps = [2, 2, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-63:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-64:
            ##cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            
            eps = [3, 3, 3] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [1, 1, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            eps = [0.5, 0.5, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            #cur, noise = cal_bfgs_rot.bfgs(params, config, c)
        elif c==-65:
            eps = [7, 7, 7] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            eps = [5, 5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            eps = [3, 3, 3] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)                                            
            eps = [1, 1, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            eps = [0.5, 0.5, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})


        elif c==-66:
            #eps = [3, 3, 3] * len(curs)
            #ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf3,
            #                            options={'maxiter': 200, 'eps': eps, 'disp': True})
            #curs = applyTranss(curs, ret.x)
            starttime = time.perf_counter()
            eps = [2, 2, 2] * len(curs)
            name = "-66 bfgs trans ngi reduced noise 1"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            ml(name, starttime, ret)                                            
            starttime = time.perf_counter()
            eps = [1, 1, 1] * len(curs)
            name = "-66 bfgs trans ngi reduced noise 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            ml(name, starttime, ret)
            starttime = time.perf_counter()
            eps = [0.5, 0.5, 0.5] * len(curs)
            name = "-66 bfgs trans ngi reduced noise 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            ml(name, starttime, ret)
        elif c==-67:
            #eps = [3, 3, 3] * len(curs)
            #ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf3,
            #                            options={'maxiter': 200, 'eps': eps, 'disp': True})
            #curs = applyTranss(curs, ret.x)
            starttime = time.perf_counter
            eps = [2, 2, 2] * len(curs)
            name = "-67 bfgs trans ngi 1"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)                                            
            ml(name, starttime, ret)
            starttime = time.perf_counter()
            eps = [1, 1, 1] * len(curs)
            name = "-67 bfgs trans ngi 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            curs = applyTranss(curs, ret.x)
            ml(name, starttime, ret)
            starttime = time.perf_counter
            name = "-67 bfgs trans ngi 3"
            eps = [0.5, 0.5, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, f, (curs,eps,config)),
                                        options={'maxiter': 200, 'eps': eps, 'disp': True})
            ml(name, starttime, ret)
        else:
            print("no method selected", c)
    
    res = applyTranss(curs, ret.x)
    #angles_noise += np.array(ret.x).reshape(angles_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return np.array(res), (trans_noise, angles_noise)
