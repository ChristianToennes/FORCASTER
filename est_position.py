import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from utils import bcolors, applyRot, applyTrans, default_config, filt_conf
from utils import minimize_log as ml
from feature_matching import *
from simple_cal import *
from objectives import *

def simulate_projections(in_cur, Ax):
    config = dict(default_config)

    primary = np.linspace(0, 360, 45, True)
    #primary = [0]
    secondary = np.linspace(0, 180, 45, True)
    #secondary = [0]
    #tertiary = np.linspace(0, 180, 90, True)
    tertiary = [0]

    bp = 0
    bs = 0
    bt = 0

    cur = np.array([[1,0,0],[0,1,0],[0,0,1]])
    cur = np.array(in_cur)
    curs = []
    pos = []
    for p in primary:
        for s in secondary:
            for t in tertiary:
                dcur = applyRot(cur, p, s, t)
                curs.append(dcur)
                pos.append([p, s, t])

    pos = np.array(pos)
    curs = np.array(curs)
    
    projs = Projection_Preprocessing(Ax(curs))

    points = [normalize_points(findInitialFeatures(projs[:,i], config), projs[:,i]) for i in range(projs.shape[1])]

    return points, pos

def est_positions(in_cur, Ax, real_imgs):
    config = dict(default_config)

    sim_points, positions = simulate_projections(in_cur, Ax)

    real_points = [normalize_points(findInitialFeatures(real_imgs[i], config), real_imgs[i]) for i in range(real_imgs.shape[0])]

    res = []

    for rp in real_points:
        cur_pos = pos[0]
        cur_obj = 0
        for sp, pos in zip(sim_points, positions):
            points, valid = matchFeatures(rp, sp)
            obj = np.count_nonzero(valid)
            if obj > cur_obj:
                cur_obj = obj
                cur_pos = pos
        res.append(applyRot(in_cur, cur_pos[0], cur_pos[1], cur_pos[2]))
    
    return res