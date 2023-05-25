import numpy as np
from utils import applyRot, default_config
from feature_matching import findInitialFeatures, normalize_points, matchFeatures, Projection_Preprocessing
from config import pdim, sdim, tdim

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
    points = []
    descs = []
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
            (p,d) = findInitialFeatures(proj, config)
            points.append(p)
            descs.append(d)
        
    pos = np.array(pos)
    return pos, points, descs


def est_position(in_cur, Ax, real_imgs, est_data):
    cur = np.array(in_cur)
    config = dict(default_config)

    pos, points, descs = est_data
    curs = []
    poss = []
    vmax = 0
    index = (0,0,0)
    for real_img in real_imgs:
        data_real = findInitialFeatures(real_img, config)
        points_real = normalize_points(data_real[0], real_img)
        no_valid = []
        indexes = []
        for i in range(0, pdim, 4):
            for j in range(0, sdim, 4):
                for k in range(0, tdim, 1):
                    idx = k*pdim*sdim+i*sdim+j
                    (p,v) = matchFeatures(data_real, (points[idx], descs[idx]), config={"lowe_ratio": 0.78}) # sin: 0.78 arc: 0.76
                    valid = np.count_nonzero(v==1)
                    no_valid.append(valid)
                    indexes.append((i,j,k))
                    if vmax==0 or valid > vmax:
                        vmax = valid
                        index = (i,j,k)
        index2 = index
        indexes = np.array(indexes)
        vmax = 0
        indexes2 = set()
        for p in np.argsort(no_valid)[-5:]:
            index = indexes[p]
            for i in range(index[0]-4,index[0]+5,1):
                if i < 0 or i >= pdim: continue
                for j in range(index[1]-4,index[1]+5,1):
                    if j < 0 or j >= sdim: continue
                    for k in range(index[2]-4,index[2]+5,1):
                        if k<0 or k>=tdim: continue
                        indexes2.add((i,j,k))
        for (i,j,k) in indexes2:
            (p,v) = matchFeatures(data_real, (points[k*sdim*pdim+i*sdim+j], descs[k*sdim*pdim+i*sdim+j]), config={"lowe_ratio": 0.75}) # sin: 0.75
            valid = np.count_nonzero(v==1)
            if vmax == 0 or valid > vmax:
                vmax = valid
                index2 = (i,j,k)

        index = index2

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

        poss.append(np.array([bp,bs,bt]))
        
        curs.append(applyRot(in_cur, poss[-1][0], poss[-1][1], poss[-1][2]))
    return np.array(curs), np.array(poss)