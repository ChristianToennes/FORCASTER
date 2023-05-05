import numpy as np
from feature_matching import *
from objectives import calcPointsObjective
from utils import applyRot

def correctXY(in_cur, config):
    cur = np.array(in_cur)

    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    points_real, features_real = data_real
    real_img = Projection_Preprocessing(real_img)

    its = 3
    if "it" in config:
        its = config["it"]
    for i in range(its):
        projs = Projection_Preprocessing(Ax(np.array([cur])))
        points,v = trackFeatures(projs[:,0], data_real, config)
        valid = v==1
        
        diff = points[valid]-points_real[valid]
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
    real_img = Projection_Preprocessing(real_img)
    its = 3
    if "it" in config:
        its = config["it"]
    for i in range(its):
        projs = Projection_Preprocessing(Ax(np.array([cur])))
        points,v = trackFeatures(projs[:,0], data_real, config)
        valid = v==1
    
        dist_new = np.sqrt(np.sum((points[valid]-points[valid])**2, axis=1))
        dist_real = np.sqrt(np.sum((points_real[valid]-points_real[valid])**2, axis=1))
        #dist_new = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in points[valid] for r in points[valid]])
        #dist_real = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in points_real[valid] for r in points_real[valid]])
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

    curs = np.array([np.array(in_cur), applyRot(in_cur, 0, 0, 180), applyRot(in_cur, 0, 180, 0), applyRot(in_cur, 180, 0, 0)])
    #curs = np.array([np.array(in_cur), applyRot(in_cur, 180, 0, 0)])
    projs = Projection_Preprocessing(Ax(curs))

    #print(projs.shape)
    features = [trackFeatures(projs[:,i], data_real, config) for i in range(projs.shape[1])]

    points_real, features_real = data_real
    #real_img = Projection_Preprocessing(real_img)

    values = np.array([calcPointsObjective(-6, points[v], points_real[v]) for i,(points,v) in enumerate(features)])

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
    real_img = Projection_Preprocessing(real_img)

    its = 3
    bt = 0
    if "it" in config:
        its = config["it"]
    for i in range(its):
        projs = Projection_Preprocessing(Ax(np.array([cur])))
        points,v = trackFeatures(projs[:,0], data_real, config)
        valid = v==1
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

        print(angle)
        #print(np.min(angle), np.mean(angle), np.median(angle), np.max(angle))
        #print(np.min(angle_cos), np.mean(angle_cos), np.median(angle_cos), np.max(angle_cos))
        projs = Projection_Preprocessing(Ax(np.array([applyRot(cur, 0,0,-np.median(angle)), applyRot(cur, 0,0,np.median(angle))])))
        points,v = trackFeatures(projs[:,0], data_real, config)
        valid = v==1
        #print(i, np.count_nonzero(valid))
        diffn = points[valid] - points_real[valid]
        points,v = trackFeatures(projs[:,1], data_real, config)
        valid = v==1
        #print(i, np.count_nonzero(valid))
        diffp = points[valid] - points_real[valid]

        #print(len(diffn), np.sum(np.abs(diffn)), len(diffp), np.sum(np.abs(diffp)))
        if np.sum(np.abs(diffn)) < np.sum(np.abs(diffp)):
            cur = applyRot(cur, 0, 0, -np.median(angle))
            bt = -np.median(angle)
        else:
            cur = applyRot(cur, 0, 0, np.median(angle))
            bt = np.median(angle)
    return cur
