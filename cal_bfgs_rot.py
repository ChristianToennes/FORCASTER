import numpy as np
import utils
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import sys
import itertools
import time
import datetime
import multiprocessing as mp
import queue
from utils import minimize_log as ml
from utils import bcolors, applyRot, applyTrans, default_config, filt_conf
from feature_matching import *
from simple_cal import *
from objectives import *
from skimage.metrics import structural_similarity,normalized_root_mse

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

def applyRots(curs, rots):
    res = []
    for i, cur in enumerate(curs):
        cur_rot = applyRot(cur, rots[i*3], rots[i*3+1], rots[i*3+2])
        res.append(cur_rot)
    return np.array(res)

def bfgs(curs, reg_config, c):
    print("bfgs rot all", c)
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
            pos = i*3
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(cur_rot)
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
        print(datetime.datetime.now(), "error", ret)
        return ret

    def f(x, curs, eps, config):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(cur_rot)
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
            cur_x = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            dvec.append(cur_x)
            dvec.append(applyRot(cur_x, eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[pos+2]))
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
            cur_x = applyRot(cur, x[pos], x[pos+1], x[pos+2])
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

        config_callback = dict(config)
        config_callback["my"] = False
        #curs = correctAll_MP(curs, config)

        if c == -30:
            config["comps"] = [(-1,1),(-2,1),(-8,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            eps = [0.1, 0.1, 0.1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            eps = [0.025, 0.025, 0.025] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
        elif c == -31:
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
        elif c == -32:
            config["comps"] = [(-1,1),(-2,1),(-8,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
        elif c == -33:
            curs = correctAll_MP(curs, config)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
        elif c==-34:
            starttime = time.perf_counter()
            name = "-34.err bfgs mixed my 1"
            eps = [0.25, 0.25, 0.25] * len(curs)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            utils.minimize_callback(name, e, (curs,eps,config), True)(np.array([0,0,0] * len(curs)))
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            #curs = correctAll_MP(curs, config)
            #utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))
            #starttime = time.perf_counter()
            #eps = [0.1, 0.1, 0.1] * len(curs)
            #name = "-34.ngi bfgs mixed my 2"
            #ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
            #                            bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
            #                            options={'maxiter': 20, 'eps': eps, 'disp': True})
            #curs = applyRots(curs, ret.x)
            #ml(name, starttime, ret)

            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))

            starttime = time.perf_counter()
            eps = [0.25, 0.25, 0.25] * len(curs)
            name = "-34.err bfgs mixed my 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            ml(name, starttime, ret)

            starttime = time.perf_counter()
            eps = [0.025, 0.025, 0.025] * len(curs)
            name = "-34.err bfgs mixed my 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            ml(name, starttime, ret)
        elif c==-35:
            starttime = time.perf_counter()
            name = "-35.err bfgs mixed my 1"
            eps = [0.25, 0.25, 0.25] * len(curs)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)), True)
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))

            starttime = time.perf_counter()
            eps = [0.05, 0.05, 0.05] * len(curs)
            name = "-35.err bfgs mixed my 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))

            starttime = time.perf_counter()
            eps = [0.01, 0.01, 0.01] * len(curs)
            name = "-35.err bfgs mixed my 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))
            ml(name, starttime, ret)
        elif c==-36:
            config["comps"] = [(-1,1),(-2,1),(-8,1)]
            curs = correctAll_MP(curs, config)
            eps = [0.1, 0.1, 0.1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
        elif c==-37:
            curs = correctAll_MP(curs, config)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            #config["comps"] = [(-6,1),(-6,1),(-6,1)]
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
        else:
            print("no method selected", c)
    else:
        
        if c==-22:
            eps = [2, 2, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-23:
            eps = [2, 2, 2, 2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-24:
            starttime = time.perf_counter()
            name = "-24.err bfgs mixed ngi 1"
            eps = [0.25, 0.25, 0.25] * len(curs)
            utils.minimize_callback(name, e, (curs,eps,config), True)(np.array([0,0,0] * len(curs)))
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
        
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            starttime = time.perf_counter()
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))

            eps = [0.05, 0.05, 0.05] * len(curs)
            name = "-24.err bfgs mixed ngi 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            starttime = time.perf_counter()
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))

            eps = [0.01, 0.01, 0.01] * len(curs)
            name = "-24.err bfgs mixed ngi 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            
            ml(name, starttime, ret)
        elif c==-25:
            curs = correctAll_MP(curs, config)
            eps = [0.25, 0.25, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            eps = [0.05, 0.05, 0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            eps = [0.01, 0.01, 0.01] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)

        elif c==-26:
            eps = [2, 2, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps,config), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-27:
            curs = correctAll_MP(curs, config)
            eps = [0.25,0.25,0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
            eps = [0.05,0.05,0.05] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
            eps = [0.01,0.01,0.01] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)

        else:
            print("no method selected", c)
    
    res = applyRots(curs, ret.x)
    angles_noise += np.array(ret.x).reshape(angles_noise.shape)

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
    print("bfgs rot all", c)
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
        config["data_real"] = np.array(real_data)
        data_real = config["data_real"]
        print(data_real.shape, real_img.shape)
        config["points_real"] = [normalize_points(data_real[i,0], real_img[i]) for i in range(data_real.shape[0])]
        config_callback = dict(config)
        config_callback["my"] = False
        config["comps"] = [(-3,1),(-4,1),(-6,1),(-3,1),(-4,1),(-6,1)]
        #curs = correctAll_MP(curs, config)

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
            pos = i*3
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(cur_rot)
        cur_x = np.array(cur_x)
        projs = Projection_Preprocessing(Ax(cur_x))
        
        for i in range(projs.shape[1]):
            q_in.put((method,(i,i,2,projs[:,i])))

        for _ in range(projs.shape[1]):
            _, _, _, r = q_out.get()
            ret += r
        
        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret#/len(curs)

    def gradf(x, curs, eps, config):
        perftime = time.perf_counter() # 150 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            dvec.append(cur_x)
            dvec.append(applyRot(cur_x, eps[pos], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[pos+1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[pos+2]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        objs = np.zeros(len(curs)*6, dtype=float)
        
        for i in range(len(curs)):
            for k in range(3):
                j = i*6+k*2
                q_in.put((method,(i,j,k,projs[:,i*4])))
                j = i*6+k*2+1
                q_in.put((method,(i,j,k,projs[:,i*4+k])))

        for _ in range(projs.shape[1]):
            i, j, k, r = q_out.get()
            objs[j] = r
            
        ret = np.zeros(len(curs)*3, dtype=float)
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
            pos = i*3
            cur_x = applyRot(cur, x[pos], x[pos+1], x[pos+2])
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
            
        ret = np.zeros(len(curs)*3, dtype=float)
        for i in range(len(ret)):
            h1 = objs[i*4]
            h_1 = objs[i*4 + 1]
            h2 = objs[i*4 + 2]
            h_2 = objs[i*4 + 3]
            ret[i] = (-h2+8*h1-8*h_1+h_2)/12
        #print("grad", time.perf_counter()-perftime)
        return ret

    if config["my"]:

        if c==-34:
            starttime = time.perf_counter()
            name = "-34.err bfgs mixed my 1"
            eps = [0.25, 0.25, 0.25] * len(curs)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            utils.minimize_callback(name, e, (curs,eps,config), True)(np.array([0,0,0] * len(curs)))
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            #curs = correctAll_MP(curs, config)
            #utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))
            #starttime = time.perf_counter()
            #eps = [0.1, 0.1, 0.1] * len(curs)
            #name = "-34.ngi bfgs mixed my 2"
            #ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
            #                            jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
            #                            bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
            #                            options={'maxiter': 20, 'eps': eps, 'disp': True})
            #curs = applyRots(curs, ret.x)
            #ml(name, starttime, ret)

            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))

            starttime = time.perf_counter()
            eps = [0.25, 0.25, 0.25] * len(curs)
            name = "-34.err bfgs mixed my 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            ml(name, starttime, ret)

            starttime = time.perf_counter()
            eps = [0.025, 0.025, 0.025] * len(curs)
            name = "-34.err bfgs mixed my 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            ml(name, starttime, ret)
        elif c==-35:
            starttime = time.perf_counter()
            name = "-35.err bfgs mixed my 1"
            eps = [0.25, 0.25, 0.25] * len(curs)
            config["comps"] = [(-3,1),(-4,1),(-6,1)]
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)), True)
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback)),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))

            starttime = time.perf_counter()
            eps = [0.05, 0.05, 0.05] * len(curs)
            name = "-35.err bfgs mixed my 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))

            starttime = time.perf_counter()
            eps = [0.01, 0.01, 0.01] * len(curs)
            name = "-35.err bfgs mixed my 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf3, callback=utils.minimize_callback(name, e, (curs,eps,config_callback), True),
                                        bounds=[(-2,2),(-2,2),(-2,2)]*len(curs),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config_callback))(np.array([0,0,0] * len(curs)))
            ml(name, starttime, ret)
        else:
            print("no method selected", c)
    else:
        
        if c==-24:
            starttime = time.perf_counter()
            name = "-24.err bfgs mixed ngi 1"
            eps = [0.25, 0.25, 0.25] * len(curs)
            utils.minimize_callback(name, e, (curs,eps,config), True)(np.array([0,0,0] * len(curs)))
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
        
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, e, (curs,eps,config)),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            starttime = time.perf_counter()
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))

            eps = [0.05, 0.05, 0.05] * len(curs)
            name = "-24.err bfgs mixed ngi 2"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            ml(name, starttime, ret)

            starttime = time.perf_counter()
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))

            eps = [0.01, 0.01, 0.01] * len(curs)
            name = "-24.err bfgs mixed ngi 3"
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps,config), method='BFGS',
                                        jac=gradf, callback=utils.minimize_callback(name, e, (curs,eps,config), True),
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})
            curs = applyRots(curs, ret.x)
            config["it"] = 1
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            curs = correctAll_MP(curs, config)
            utils.minimize_callback(name, e, (curs,eps,config))(np.array([0,0,0] * len(curs)))
            
            ml(name, starttime, ret)
        else:
            print("no method selected", c)
    
    res = applyRots(curs, ret.x)
    angles_noise += np.array(ret.x).reshape(angles_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    for t in ts:
        q_in.put(("exit", None))

    return np.array(res), (trans_noise, angles_noise)
