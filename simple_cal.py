import numpy as np
from feature_matching import *
from objectives import calcPointsObjective
from utils import applyRot, Ax_param_asta, default_config
import multiprocessing as mp
import sys
import io
import matplotlib.pyplot as plt

def correctXY(in_cur, config):
    cur = np.array(in_cur)

    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)

    its = 3
    if "it" in config:
        its = config["it"]
    for i in range(its):
        projs = Projection_Preprocessing(Ax(np.array([cur])))
        p,v = trackFeatures(projs[:,0], data_real, config)
        points = normalize_points(p, projs[:,0])
        valid = v==1
    
        points = points[valid]
        
        diff = np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points,points_real[valid])])
        if len(diff)>1:
            
            xdir = cur[1]#/np.linalg.norm(cur[1])
            ydir = cur[2]#/np.linalg.norm(cur[2])
            #m = np.mean(diff, axis=0)
            #std = np.std(diff, axis=0)
            #diff = diff[np.bitwise_and(np.bitwise_and(diff[:,0]>m[0]-3*std[0], diff[:,0]<m[0]+3*std[0]), np.bitwise_and(diff[:,1]>m[1]-3*std[1], diff[:,1]<m[1]+3*std[1]))]
            if "mean" in config and config["mean"]:
                med = np.mean(diff, axis=0)
            else:
                med = np.median(diff, axis=0)
            #print(m, std, med, med[0]*xdir, med[1]*ydir)
            cur[0] += med[0] * xdir# / np.linalg.norm(xdir)
            cur[0] += med[1] * ydir# / np.linalg.norm(ydir)
            if (np.abs(med[0]*xdir) > 200).any() or (np.abs(med[1]*ydir) > 200).any():
                #print(in_cur, cur)
                #print(m, std)
                print(med[0], xdir)
                print(med[1], ydir)
                raise Exception("xy change to large")
            if (np.abs(med[0]*xdir) < 0.1).all() and (np.abs(med[1]*ydir) < 0.1).all():
                break
    return cur

def correctZ(in_cur, config):
    #print("z", end=" ")
    cur = np.array(in_cur)

    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)
    its = 3
    if "it" in config:
        its = config["it"]
    for i in range(its):
        projs = Projection_Preprocessing(Ax(np.array([cur])))
        p,v = trackFeatures(projs[:,0], data_real, config)
        points = normalize_points(p, projs[:,0])
        valid = v==1

        #points = points[valid]
    
        dist_new = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in points[valid] for r in points[valid]])
        dist_real = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in points_real[valid] for r in points_real[valid]])
        if np.count_nonzero(dist_new) > 5:
            if "mean" in config and config["mean"]:
                scale = Ax.distance_source_origin*(np.mean(dist_real[dist_new!=0]/dist_new[dist_new!=0])-1)
            else:
                scale = Ax.distance_source_origin*(np.median(dist_real[dist_new!=0]/dist_new[dist_new!=0])-1)
            zdir = np.cross(cur[2], cur[1])
            zdir = zdir / np.linalg.norm(zdir)
            cur[0] += scale * zdir
            if np.linalg.norm(cur[0]-in_cur[0])>500:
                print("large z", end=" ")
                #print(scale, zdir)
                #print(np.median(dist_real[dist_new!=0]/dist_new[dist_new!=0]), np.mean(dist_real[dist_new!=0]/dist_new[dist_new!=0]))
                #print(dist_new[dist_new!=0].shape, np.min(dist_new[dist_new!=0]), np.min(np.abs(dist_new[dist_new!=0])))
                #raise Exception("z change to large")
                return np.array(in_cur)
            if (np.abs(scale*zdir)<0.1).all():
                break
        else:
            cur = np.array(in_cur)
    return cur

def correctFlip(in_cur, config):
    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    curs = np.array([np.array(in_cur), applyRot(in_cur, 0, 0, 180), applyRot(in_cur, 180, 0, 180), applyRot(in_cur, 180, 0, 0)])
    #curs = np.array([np.array(in_cur), applyRot(in_cur, 180, 0, 0)])
    projs = Projection_Preprocessing(Ax(curs))

    #print(projs.shape)
    features = [trackFeatures(projs[:,i], data_real, config) for i in range(projs.shape[1])]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    #real_img = Projection_Preprocessing(real_img)

    values = np.array([calcPointsObjective(-6, normalize_points(points[v], projs[:,i]), points_real[v]) for i,(points,v) in enumerate(features)])

    return curs[np.argmin(values)]

def correctTrans(cur, config):
    #config["it"] = 3
    cur = correctXY(cur, config)
    cur = correctZ(cur, config)
    cur = correctXY(cur, config)
    return cur

def correctRotZ(in_cur, config): 
    cur = np.array(in_cur)

    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)

    its = 3
    bt = 0
    if "it" in config:
        its = config["it"]
    for i in range(its):
        projs = Projection_Preprocessing(Ax(np.array([cur])))
        p,v = trackFeatures(projs[:,0], data_real, config)
        points = normalize_points(p, projs[:,0])
        valid = v==1
        #print(i, np.count_nonzero(valid))
    
        points = points[valid]
        
        points_r = points_real[valid]
        try:
            new_mid = np.mean(points, axis=0)
            real_mid = np.mean(points_r, axis=0)
        except Exception as e:
            print(e)
            return in_cur

        points = points - new_mid
        points_r = points_r- real_mid

        #c = np.linalg.norm(points_new-points_real, axis=-1)
        #a = np.linalg.norm(points, axis=-1)
        #b = np.linalg.norm(points_r, axis=-1)
        #print(a.shape, b.shape, points_new.shape)

        #angle = np.arccos((a*a+b*b-c*c) / (2*a*b))*180.0/np.pi
        #angle_cos = np.arccos( (points[:,0]*points_r[:,0]+points[:,1]*points_r[:,1]) / (a*b) )*180.0/np.pi
        angle = (np.arctan2(points[:,0], points[:,1])-np.arctan2(points_r[:,0], points_r[:,1])) * 180.0/np.pi
        angle[angle<-180] += 360
        angle[angle>180] -= 360

        #print(np.min(angle), np.mean(angle), np.median(angle), np.max(angle))
        #print(np.min(angle_cos), np.mean(angle_cos), np.median(angle_cos), np.max(angle_cos))
        projs = Projection_Preprocessing(Ax(np.array([applyRot(cur, 0,0,-np.median(angle)), applyRot(cur, 0,0,np.median(angle))])))
        p,v = trackFeatures(projs[:,0], data_real, config)
        points = normalize_points(p, projs[:,0])
        valid = v==1
        #print(i, np.count_nonzero(valid))
        diffn = np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])
        p,v = trackFeatures(projs[:,1], data_real, config)
        points = normalize_points(p, projs[:,1])
        valid = v==1
        #print(i, np.count_nonzero(valid))
        diffp = np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points[valid],points_real[valid])])

        #print(len(diffn), np.sum(np.abs(diffn)), len(diffp), np.sum(np.abs(diffp)))
        if np.sum(np.abs(diffn)) < np.sum(np.abs(diffp)):
            cur = applyRot(cur, 0, 0, -np.median(angle))
            bt = -np.median(angle)
        else:
            cur = applyRot(cur, 0, 0, np.median(angle))
            bt = np.median(angle)
    return cur

def correctAll(curs, config):
    out_curs = []
    for i, cur in enumerate(curs):
        conf = dict(config)
        conf["data_real"] = config["data_real"][i]
        conf["real_img"] = config["real_img"][i]
        out_curs.append(correctTrans(cur, conf))
    return np.array(out_curs)

def it_func(con, Ax_params, ready, name):
    try:
        #print("start")
        np.seterr(all='raise')
        Ax = None
        Ax = Ax_param_asta(*Ax_params)
        while True:
            try:
                con.send(("ready",))
                ready.set()
                (i, cur, im, noise, data_real) = con.recv()
                #print(i)
                old_stdout = sys.stdout
                sys.stdout = stringout = io.StringIO()

                real_img = Projection_Preprocessing(im)
                cur_config = dict(default_config)
                cur_config["real_img"] = real_img
                cur_config["Ax"] = Ax
                cur_config["noise"] = noise
                data_real = findInitialFeatures(real_img, cur_config)
                cur_config["data_real"] =  data_real
                try:
                    cur = correctTrans(cur, cur_config)
                except Exception as ex:
                    print(ex, i, cur, file=sys.stderr)
                stringout.flush()
                con.send(("result",i,cur,stringout.getvalue()))
                ready.set()
                stringout.close()
                sys.stdout = old_stdout
            except EOFError:
                break
            except BrokenPipeError:
                return
        try:
            con.send(("error",))
        except EOFError:
            pass
        except BrokenPipeError:
            pass
    except KeyboardInterrupt:
        pass

def correctAll_MP(curs, config):
    corrs = []
    pool_size = mp.cpu_count()-1
    
    pool = []
    proc_count = 0
    ready = mp.Event()
    
    corrs = np.array([None]*len(curs))
    indices = list(range(len(curs)))

    while np.array([e is None for e in corrs]).any(): #len(indices)>0:
        ready_con = None
        while ready_con is None:
            for _ in range(len(pool), pool_size):
                p = mp.Pipe(True)
                if "profile" in config and config["profile"]:
                    name = config["name"]+"_"+str(proc_count)
                else:
                    name = None
                proc = mp.Process(target=it_func, args=(p[1], config["Ax_gen"], ready, name), daemon=True)
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
            ready_con[0].send((i, curs[i], config["real_img"][i], (config["noise"][0][i],config["noise"][1][i]), None))
            ready_con[3] = i

    for con in pool:
        con[2].terminate()
        con[0].close()
        con[1].close()
        
    corrs = np.array(corrs.tolist())
    print()
    #print(corrs)
    return corrs