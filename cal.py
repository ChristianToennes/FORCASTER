import numpy as np
import cv2
from scipy.interpolate import ndgriddata
import utils
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
import sys
import itertools
import time
import threading
import queue

default_config = {"use_cpu": True, "AKAZE_params": {"threshold": 0.0005, "nOctaves": 4, "nOctaveLayers": 4},
"my": True, "grad_width": (1,25), "noise": None, "both": False, "max_change": 1}

def applyRot(in_params, α, β, γ):
    params = np.array(in_params)
    #print(params, α, β, γ)
    if α!=0:
        params[1] = utils.rotMat(α, params[2]).dot(params[1])
    if β!=0:
        params[2] = utils.rotMat(β, params[1]).dot(params[2])
    if γ!=0:
        u = np.cross(params[1], params[2])
        R = utils.rotMat(γ, u)
        params[1] = R.dot(params[1])
        params[2] = R.dot(params[2])
    return params

def applyTrans(in_params, x, y, z):
    cur = np.array(in_params)
    if x != 0:
        xdir = cur[1]
        cur[0] += x * xdir
    if y != 0:
        ydir = cur[2]
        cur[0] += y * ydir
    if z != 0:
        zdir = np.cross(cur[2], cur[1])
        #zdir = zdir / np.linalg.norm(zdir)
        cur[0] += z * zdir
    return cur

def trackFeatures(next_img, data, config):
    #perftime = time.perf_counter()
    base_points, f1 = data
    new_points, f2 = findInitialFeatures(next_img, config)
    # FLANN parameters
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)   # or pass empty dictionary
    #matcher = cv2.FlannBasedMatcher(index_params,search_params)
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    if f1 is None:
        print("no features in old image")
    if f2 is None:
        print("no features in new image")
        return base_points, np.zeros(len(base_points), dtype=bool) 

    matches = matcher.knnMatch(f1, f2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = np.zeros((len(matches), 2))
    points = -np.ones(len(base_points), dtype=int)
    valid = np.zeros(len(base_points), dtype=bool)

    # ratio test as per Lowe's paper
    dists = []
    for i,ms in enumerate(matches):
        if len(ms) > 1:
            m,n = ms    
            dists.append(m.distance)
            if m.distance < 0.7*n.distance:
                matchesMask[i,0]=1
                valid[m.queryIdx] = True
                points[m.queryIdx] = m.trainIdx
            else:
                valid[m.queryIdx] = False
        else:
            dists.append(-1)
            valid[ms[0].queryIdx] = False

    dists = np.array(dists)
    

    for i in range(len(base_points)):
        if np.count_nonzero(points==i)>1:
            valid[points==i] = False
            points[points==i] = -1
            for i2,(m,n) in enumerate(matches):
                if m.trainIdx == i:
                    matchesMask[i2,0] = 0

    if len(dists[valid])==0:
        #real_img = config["real_img"]
        #img = cv2.drawMatchesKnn(real_img,base_points,next_img,new_points,matches,None,matchesMask=matchesMask)
        ##plt.imshow(img)
        #plt.show()
        #plt.close()
        pass
    else:
        m = np.mean(dists[valid])
        std = np.std(dists[valid])
        out = np.bitwise_or(dists<m-3*std, dists>m+3*std)
        valid[out] = False

    return new_points[points], valid

def findInitialFeatures(img, config):
    #global detector, gpu_img, mask, gpu_mask
    #perftime = time.perf_counter()
    gpu_img = None
    detector = None
    mask = None
    gpu_mask = None

    if mask is None:
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[50:-50,50:-50] = True
    
    if config["use_cpu"]:
        #detector = cv2.xfeatures2d_SURF.create(100, 4, 3, False, True)
        #detector = cv2.SIFT_create()
        detector = cv2.AKAZE_create(**config["AKAZE_params"])
        #detector = cv2.xfeatures2d.StarDetector_create(maxSize=45,responseThreshold=10,lineThresholdProjected=10,lineThresholdBinarized=8,suppressNonmaxSize=5)
        #brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        #detector = cv2.ORB_create(nfeatures=300, scaleFactor=1.4, nlevels=4, edgeThreshold=41, patchSize=41, fastThreshold=5)
        points, features = detector.detectAndCompute(img, mask)
        #points = detector.detect(img, mask)
        #points, features = brief.compute(img, points)
        points = np.array(points)
        
        #feature_params = {"maxCorners": 200, "qualityLevel": 0.01, "minDistance": 19, "blockSize": 21}
        #feature_params = {"maxCorners": 200, "qualityLevel": 0.05, "minDistance": 13, "blockSize": 17}
        #points = cv2.goodFeaturesToTrack(img, mask=mask, **feature_params)
        #points = points[:,0]
        #features = None
        #print(points.shape)
    else:
        if "feat_thres" in config:
            feat_thres = config["feat_thres"]
        else:
            feat_thres = 100
        if detector is None:
            detector = cv2.cuda.SURF_CUDA_create(feat_thres, 4, 3, False, 0.01, True)
        if gpu_img is None:
            gpu_img = cv2.cuda_GpuMat()
        if gpu_mask is None:
            gpu_mask = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_mask.upload(mask)
        while(1):
            try:
                points, features = detector.detectWithDescriptors(gpu_img, gpu_mask)
                break
            except Exception as e:
                pass
        points = points.download()
        features = features.download()
        points = np.swapaxes(points, 0,1)
        points = np.array([cv2.KeyPoint(p[0],p[1],p[4],p[5],p[6],int(p[3]),i) for i,p in enumerate(points)])
    #print("findInitialFeatures", time.perf_counter()-perftime)
    return points, features

def Projection_Preprocessing(proj, alpha=0, beta=255):
    proj = np.array(proj, dtype=np.float32)
    mean = np.mean(proj)
    std = np.std(proj)
    return (proj-mean) / std

    if len(proj.shape) == 2:
        return cv2.normalize(proj, None, alpha, beta, cv2.NORM_MINMAX).astype('uint8')
    else:
        return np.swapaxes(np.array([cv2.normalize(proj[:,i], None, alpha, beta, cv2.NORM_MINMAX).astype('uint8') for i in range(proj.shape[1])]), 0,1)

def normalize_points(points, img):
    xdim, ydim = img.shape
    if len(points.shape) == 1:
        return np.array([[1000.0*p.pt[0]/xdim, 1000.0*p.pt[1]/ydim] for p in points])
    else:
        ret = np.array(points)
        ret[...,0] *= 1000.0/xdim
        ret[...,1] *= 1000.0/ydim
        return ret

def unnormalize_points(points, img):
    xdim, ydim = img.shape
    ret = np.array(points)
    ret[:,0] *= xdim/1000.0
    ret[:,1] *= ydim/1000.0
    return ret

def GI_(old_img, new_img):
    p1di = (old_img[1:]-old_img[:-1]).flatten()
    p1dj = (old_img[:,1:]-old_img[:,:-1]).flatten()
    p2di = (new_img[1:]-new_img[:-1]).flatten()
    p2dj = (new_img[:,1:]-new_img[:,:-1]).flatten()
    ret = 0
    for p1 in itertools.product(p1di,p1dj):
        absp1 = np.linalg.norm(p1)
        for p2 in itertools.product(p2di,p2dj):
            absp2 = np.linalg.norm(p2)
            absGrad = absp1*absp2
            if absGrad == 0:
                continue
            gradDot = p1[0]*p2[0] + p1[1]*p2[1]
            w = 0.5*(gradDot / absGrad + 1)
            ret += w * np.min(absp1, absp2)
    return ret

def GI__(old_img, new_img):
    s = 1
    p1 = np.meshgrid(old_img[1::s,::s]-old_img[:-1:s,::s], old_img[::s,1::s]-old_img[::s,:-1:s], copy=False)
    p2 = np.meshgrid(new_img[1::s,::s]-new_img[:-1:s,::s], new_img[::s,1::s]-new_img[::s,:-1:s], copy=False)
    absp1 = np.linalg.norm([p1[0].flatten(),p1[1].flatten()], axis=-1)
    absp2 = np.linalg.norm([p2[0].flatten(),p2[1].flatten()], axis=-1)
    
    absGrad = absp1*absp2
    gradDot = p1[0].flatten()*p2[0].flatten() - p1[1].flatten()*p2[1].flatten()
    gradDot[absGrad==0] = 0
    absGrad[absGrad==0] = 1
    w = 0.5*(gradDot / absGrad + 1)
    
    return np.sum(w*np.min(np.array([absp1, absp2]), axis=0))

gi_mask = None
gi_shape = None
def GI(new_img, p1, absp1, gi_skip=1):
    p2 = new_img[1::gi_skip,:-1:gi_skip]-new_img[:-1:gi_skip,:-1:gi_skip], new_img[:-1:gi_skip,1::gi_skip]-new_img[:-1:gi_skip,:-1:gi_skip]
    absp2 = np.sqrt(p2[0]*p2[0] + p2[1]*p2[1], dtype=np.float32)
    #absp2 = np.linalg.norm(p2, ord=2, axis=0)
    absGrad = absp1[::gi_skip, ::gi_skip]*absp2
    minabs = np.min(np.array([absp1[::gi_skip, ::gi_skip], absp2]), axis=0)
    #del absp2
    gradDot = p1[0][::gi_skip, ::gi_skip]*p2[0] + p1[1][::gi_skip, ::gi_skip]*p2[1]
    #del p2
    f = absGrad!=0
    gradDot = gradDot*f
    absGrad[~f] = 1
    
    w = 0.5*(gradDot / absGrad + 1)
    #del gradDot
    #del absGrad
    r = w*minabs
    #del w
    ret = np.sum(r)
    #del r
    return ret

def _GI(old_img, new_img):
    p1 = np.array([(old_img[1:,:-1]-old_img[:-1,:-1]).flatten(),(old_img[:-1,1:]-old_img[:-1,:-1]).flatten()]).T
    p2 = np.array([(new_img[1:,:-1]-new_img[:-1,:-1]).flatten(),(new_img[:-1,1:]-new_img[:-1,:-1]).flatten()]).T
    absp1 = np.linalg.norm(p1, axis=-1)
    absp2 = np.linalg.norm(p2, axis=-1)

    absGrad = absp1*absp2
    gradDot = p1[:,0]*p2[:,0] - p1[:,1]*p2[:,1]
    gradDot[absGrad==0] = 0
    absGrad[absGrad==0] = 1
    w = 0.5*(gradDot / absGrad + 1)
    print(p1.shape, p2.shape, absp1.shape, absp2.shape, w.shape)
    exit()
    return np.sum(w*np.min(np.array([absp1, absp2]), axis=0))

gis = []

def calcGIObjective(old_img_big, new_img_big, i, cur, config):
    global gi_mask, gi_shape
    if gi_mask is None or gi_mask.shape != old_img_big.shape:
        gi_mask = np.zeros_like(old_img_big, dtype=bool)
        b1 = old_img_big.shape[0]//4
        b2 = old_img_big.shape[1]//4
        gi_mask[b1:-b1,b2:-b2] = True
        gi_shape = (old_img_big.shape[0]-b1-b1, old_img_big.shape[1]-b2-b2)

    old_img = old_img_big[gi_mask].reshape(gi_shape)
    new_img = new_img_big[gi_mask].reshape(gi_shape)
    #if cur is not None:
    #    for key in gis[i].keys():
    #        k=np.array(key)
    #        if np.linalg.norm(k[0]-cur[0]) < 0.01 and np.linalg.norm(k[1]-cur[1]) < 0.01 and np.linalg.norm(k[2]-cur[2]) < 0.01:
    #            return gis[i][key]

    if config["GIoldold"][i] is None:
        p1 = old_img[1:,:-1]-old_img[:-1,:-1], old_img[:-1,1:]-old_img[:-1,:-1]
        config["p1"][i] = p1
        config["absp1"][i] = np.sqrt(p1[0]*p1[0] + p1[1]*p1[1],dtype=np.float32)
        #config["absp1"][i] = np.linalg.norm(p1, ord=2, axis=0)
        config["GIoldold"][i] = np.array([GI(old_img, p1, config["absp1"][i]), GI(old_img, p1, config["absp1"][i], 2), GI(old_img, p1, config["absp1"][i], 4)])
    #perftime = time.perf_counter()
    GIoldnew =np.array([GI(new_img, config["p1"][i], config["absp1"][i]),GI(new_img, config["p1"][i], config["absp1"][i], 2),GI(new_img, config["p1"][i], config["absp1"][i], 4)])
    #print("GI", time.perf_counter()-perftime)
    #return GIoldnew / config["GIoldold"]
    ngi = np.sum(GIoldnew / config["GIoldold"][i]) / 3
    #if cur is not None:
    #    gis[i][tuple((tuple(c) for c in cur))] = ngi
    return 1.0/(ngi+1e-10)

def calcPointsObjective(comp, good_new, good_old):
    if comp==10:
        d = good_new[:,0]-good_old[:,0]
        if len(d)==0:
            f = -1
        else:
            std = np.std(d)
            mean = np.mean(d)
            fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]
            #fd = d[np.bitwise_and(d>np.quantile(d,0.1), d<np.quantile(d,0.9))]
            #print(d.shape, fd.shape, mean, std)
            if len(fd)==0:
                f = -1
            else:
                f = np.std( fd )
    elif comp==11:
        d = good_new[:,1]-good_old[:,1]
        if len(d)==0:
            f = -1
        else:
            std = np.std(d)
            mean = np.mean(d)
            fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]
            #fd = d[np.bitwise_and(d>np.quantile(d,0.1), d<np.quantile(d,0.9))]
            if len(fd)==0:
                f = -1
            else:
                f = np.std( fd )
    elif comp==12:
        d = np.linalg.norm(good_new-good_old, axis=1)
        if len(d)==0:
            f = -1
        else:
            std = np.std(d)
            mean = np.mean(d)
            fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]
            if len(fd)==0:
                f = -1
            else:
                f = np.std(fd)
    elif comp==0:
        a = (good_new[:,0,np.newaxis]-good_new[:,0])
        b = (good_old[:,0,np.newaxis]-good_old[:,0])
        a = a[b!=0]
        b = b[b!=0]
        r = a/b - 1
        if len(r)>0:
            std = np.std(r)
            mean = np.mean(r)
            fd = r[np.bitwise_and(r<=mean+3*std, r>=mean-3*std)]
            f = np.abs( np.mean( fd ) )
        else: 
            f = -1
    elif comp==1:
        a = (good_new[:,1,np.newaxis]-good_new[:,1])
        b = (good_old[:,1,np.newaxis]-good_old[:,1])
        a = a[b!=0]
        b = b[b!=0]
        r = a / b - 1
        if len(r)>0:
            std = np.std(r)
            mean = np.mean(r)
            fd = r[np.bitwise_and(r<=mean+3*std, r>=mean-3*std)]
            f = np.abs ( np.mean( fd ) )
        else: 
            f = -1
    elif comp==2:
        #f = np.var( good_new[:,1]-good_old[:,1] )
        a = (good_new[:,np.newaxis]-good_new).reshape((-1,2))
        b = (good_old[:,np.newaxis]-good_old).reshape((-1,2))

        filt = ~np.eye(good_new.shape[0],dtype=bool).flatten()
        a = a[filt]
        b = b[filt]
        ϕ_new = np.arctan2(a[:,1], a[:,0])
        ϕ_old = np.arctan2(b[:,1], b[:,0])

        d = ϕ_new*180/np.pi-ϕ_old*180/np.pi
        d[d<-180] = -360-d[d<-180]
        d[d>180] = 360-d[d>180]

        if len(d) > 0:
            std = np.std(d)
            mean = np.mean(d)
            fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]
            f = np.abs( np.mean( fd ) )
        else:
            f = -1
    elif comp==22:
        mid_n_x, mid_n_y = np.mean(good_new, axis=0)
        mid_o_x, mid_o_y = np.mean(good_old, axis=0)
        ϕ_new = np.arctan2(good_new[:,1]-mid_n_y, good_new[:,0]-mid_n_x)
        ϕ_old = np.arctan2(good_old[:,1]-mid_o_y, good_old[:,0]-mid_o_x)

        #ϕ_new = np.array([np.arctan2(l[1]-r[1], l[0]-r[0])  for l in good_new for r in good_new if (l!=r).any()])
        #ϕ_old = np.array([np.arctan2(l[1]-r[1], l[0]-r[0])  for l in good_old for r in good_old if (l!=r).any()])

        ϕ_new[ϕ_new<-np.pi] += 2*np.pi
        ϕ_new[ϕ_new>np.pi] -= 2*np.pi

        ϕ_old[ϕ_old<-np.pi] += 2*np.pi
        ϕ_old[ϕ_old>np.pi] -= 2*np.pi

        f = np.abs((np.mean(ϕ_new)-np.mean(ϕ_old))*180/np.pi)
        if f > 180: f = 360-f
    elif comp==32:
        mid_n_x, mid_n_y = np.mean(good_new, axis=0)
        mid_o_x, mid_o_y = np.mean(good_old, axis=0)
        ϕ_new = np.arctan2(good_new[:,1]-mid_n_y, good_new[:,0]-mid_n_x)+np.pi
        ϕ_old = np.arctan2(good_old[:,1]-mid_o_y, good_old[:,0]-mid_o_x)+np.pi

        f = np.abs((np.mean(ϕ_new)-np.mean(ϕ_old))*180/np.pi)
        if f > 180: f = 360-f

    elif comp==-1:
        d = good_new[:,0]-good_old[:,0]
        if len(d)==0:
            f = -1
        else:
            std = np.std(d)
            mean = np.mean(d)
            fd = d[np.bitwise_and(d<mean+3*std, d>mean-3*std)]
            #fd = d[np.bitwise_and(d>np.quantile(d,0.1), d<np.quantile(d,0.9))]
            #print(d.shape, fd.shape, mean, std)
            if len(fd)==0:
                f = -1
            else:
                f = np.abs(np.median( fd ))
    elif comp==-2:
        d = good_new[:,1]-good_old[:,1]
        if len(d)==0:
            f = -1
        else:
            std = np.std(d)
            mean = np.mean(d)
            fd = d[np.bitwise_and(d<mean+3*std, d>mean-3*std)]
            #fd = d[np.bitwise_and(d>np.quantile(d,0.1), d<np.quantile(d,0.9))]
            if len(fd)==0:
                f = -1
            else:
                f = np.abs(np.median( fd ))
    elif comp==-3:
        dist_new = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in good_new for r in good_new])
        dist_real = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in good_old for r in good_old])
        if np.count_nonzero(dist_new) > 5:
            f = np.abs(np.median(dist_real[dist_new!=0]/dist_new[dist_new!=0])-1)
        else:
            f = -1
    
    else:
        f = -2
        
    return f

def calcMyObjective(axis, proj, config):
    data_real = config["data_real"]
    (p,v), proj = trackFeatures(proj, data_real, config), proj
    points = normalize_points(p, proj)
    valid = v==1
    points = points[valid]
    value = calcPointsObjective(axis, points, data_real[0][valid])
    return value

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def correctXY(in_cur, config):
    cur = np.array(in_cur)
    data_real = config["data_real"]
    real_img = config["real_img"]
    Ax = config["Ax"]

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)

    its = 20
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
            m = np.mean(diff, axis=0)
            std = np.std(diff, axis=0)
            diff = diff[np.bitwise_and(np.bitwise_and(diff[:,0]>m[0]-3*std[0], diff[:,0]<m[0]+3*std[0]), np.bitwise_and(diff[:,1]>m[1]-3*std[1], diff[:,1]<m[1]+3*std[1]))]
            med = np.median(diff, axis=0)*0.1
            #print(m, std, med, med[0]*xdir, med[1]*ydir)
            cur[0] += med[0] * xdir# / np.linalg.norm(xdir)
            cur[0] += med[1] * ydir# / np.linalg.norm(ydir)
            if (np.abs(med[0]*xdir) > 200).any() or (np.abs(med[1]*ydir) > 200).any():
                #print(in_cur, cur)
                print(m, std)
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
    its = 20
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

    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)
    if not my:
        config["GIoldold"] = GI(real_img, real_img)
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
        for (p,v), proj in [(trackFeatures(projs[:,i], data_real, config), projs[:,i]) for i in range(projs.shape[1])]:
            points.append(normalize_points(p, proj))
            valid.append(v==1)
        combined_valid = valid[0]
        for v in valid:
            combined_valid = np.bitwise_and(combined_valid, v)
        points = [p[v] for p, v in zip(points, valid)]
        
        #print(points.shape, points_real.shape)
        values = np.array([calcPointsObjective(axis, points, points_real[v]) for points,v in zip(points,valid)])
        #p = np.polyfit(εs, values, 2)

        #values_out = values>(np.mean(values)+3*np.std(values))
        #values[values_out] = np.median(values)

        mid = np.argmin(values)
        skip = np.count_nonzero(values<0)
        midpoints = np.argsort(values)[skip:skip+5]
        if len(midpoints) == 0:
            #print(values)
            return cur
        mean_mid = np.mean(εs[midpoints])
        if False and mid > 10 and mid < grad_width[1]*2-10:
            std_mid = np.std(εs[midpoints])
            mid = np.argmin(values[midpoints[np.bitwise_and(εs[midpoints]>mean_mid-3*std_mid, εs[midpoints]<mean_mid+3*std_mid)]])
            mid = midpoints[np.bitwise_and(εs[midpoints]>mean_mid-2*std_mid, εs[midpoints]<mean_mid+2*std_mid)][mid]
            #with warnings.catch_warnings():
            #    warnings.simplefilter("error")
            #    try:
            #        p1 = np.polyfit(εs[:mid+1], values[:mid+1], 1)
            #    except np.RankWarning:
            #        if both:
            #            return cur, 0
            #        return cur
            #with warnings.catch_warnings():
            #    warnings.simplefilter("error")
            #    try:
            #        p2 = np.polyfit(εs[mid:], values[mid:], 1)
            #    except np.RankWarning:
            #        if both:
            #            return cur, 0
            #        return cur
        #else:
            p1 = np.polyfit(εs[:mid+1], values[:mid+1], 1)
            p2 = np.polyfit(εs[mid:], values[mid:], 1)
            min_ε = np.roots(p1-p2)[0]
        else:
            min_ε = mean_mid
    else:
        values = np.array([calcGIObjective(real_img, projs[:,i], config) for i in range(projs.shape[1])])
        #p = np.polyfit(εs, values, 2)
        #mid = np.argmax(values)
        #p1 = np.polyfit(εs[:mid+1], values[:mid+1], 1)
        #p2 = np.polyfit(εs[mid:], values[mid:], 1)
    
        #min_ε = np.roots(p1-p2)[0]
        #mid = np.argmax(values)
        skip = np.count_nonzero(values<0)
        midpoints = np.argsort(values)[-5:]
        #if len(midpoints) == 0:
        #    print(values)
        mean_mid = np.mean(εs[midpoints])
        min_ε = mean_mid

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
        config["GIoldold"] = GI(real_img, real_img)

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
            values = np.array([1-calcGIObjective(real_img, projs[:,i], config) for i in range(projs.shape[1])])
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


def roughRegistration(in_cur, reg_config, c):
    cur = np.array(in_cur)
    config = dict(default_config)
    config.update(reg_config)
    if c==-19:
        config["opti_method"] = "Newton-CG"
        return bfgs(cur, config, 1)
    if c==-18:
        config["opti_method"] = "TNC"
        return bfgs(cur, config, 1)
    if c==-17:
        config["opti_method"] = "trust-exact"
        return bfgs(cur, config, 1)
    if c==-16:
        config["opti_method"] = "trust-krylov"
        return bfgs(cur, config, 1)
    if c==-15:
        config["opti_method"] = "trust-ncg"
        return bfgs(cur, config, 1)
    if c==-14:
        config["opti_method"] = "dogleg"
        return bfgs(cur, config, 1)
    if c==-13:
        config["opti_method"] = "SLSQP"
        return bfgs(cur, config, 1)
    if c==-12:
        config["opti_method"] = "COBYLA"
        return bfgs(cur, config, 1)
    if c==-7:
        config["opti_method"] = "trust-exact"
        return bfgs(cur, config, 0)
    if c==-6:
        config["opti_method"] = "trust-krylov"
        return bfgs(cur, config, 0)
    if c==-5:
        config["opti_method"] = "trust-ncg"
        return bfgs(cur, config, 0)
    if c==-4:
        config["opti_method"] = "dogleg"
        return bfgs(cur, config, 0)
    if c==-3:
        config["opti_method"] = "SLSQP"
        return bfgs(cur, config, 0)
    if c==-2:
        config["opti_method"] = "COBYLA"
        return bfgs(cur, config, 0)
    if c==-1:
        return bfgs(cur, reg_config, 0)
    if c==0:
        return bfgs(cur, reg_config, 1)
    if c==1 or c==2 or c<=-20:
        return bfgs_trans(cur, reg_config, c)
    config = dict(default_config)
    config.update(reg_config)

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    #print("rough")
    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)
    
    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)
    
    #plt.figure()
    #plt.imshow(real_img)
    #plt.figure()
    #plt.imshow(proj_img)
    #plt.show()
    #plt.close()

    def grad_img(img):
        i = img[1:]-img[:-1]
        i = i[:,1:]-i[:,:-1]
        return i

    def debug_projs(cur):
        plt.figure()
        plt.title("real")
        plt.imshow(grad_img(real_img))
        plt.figure()
        plt.title("cur_proj")
        cur_proj = Projection_Preprocessing(Ax(np.array([cur]))[:,0])
        plt.imshow(grad_img(cur_proj))
        plt.figure()
        plt.title("diff_prev")
        plt.imshow(grad_img(real_img)-grad_img(proj_img))
        plt.figure()
        plt.title("diff_cur")
        plt.imshow(grad_img(real_img)-grad_img(cur_proj))
        plt.figure()
        plt.title("diff_proj")
        plt.imshow(grad_img(proj_img)-grad_img(cur_proj))
        plt.show()
        plt.close()

    if c==3: # 3
        config["it"] = 1
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==4: # 4
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c == 5:
        config["it"] = 10
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c == 7:
        config["it"] = 1
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
    elif c==8:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
    elif c==9:
        config["it"] = 10
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
    elif c==10:
        config["it"] = 1
        cur = correctXY(cur, config)
    elif c==11:
        config["it"] = 3
        cur = correctXY(cur, config)
    elif c==12:
        config["it"] = 5
        cur = correctXY(cur, config)
    elif c==13:
        config["it"] = 10
        cur = correctXY(cur, config)
    elif c==15: # 5
        config["my"] = False
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
                cur = correctXY(cur, config)
        
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
    elif c==25:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        for grad_width in [(2.5,25), (0.5,15)]:
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
    elif c==26:
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        for grad_width in [(2.5,25), (0.5,25)]:
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

    elif c==30:
        config["my"] = False
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
    elif c==31:
        config["my"] = False
        config["it"] = 3
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)
        cur = correctZ(cur, config)
        cur = correctXY(cur, config)

        config["it"] = 3
        for grad_width in [(2.5,25), (0.5,25)]:
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

    return cur

def correctTrans(cur, config):
    config["it"] = 3
    cur = correctXY(cur, config)
    cur = correctZ(cur, config)
    cur = correctXY(cur, config)
    cur = correctZ(cur, config)
    cur = correctXY(cur, config)
    return cur

def bfgs(cur, reg_config, c):
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c==1

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)

    if 'opti_method' not in config:
        config['opti_method'] = 'L-BFGS-B'
    #else:
    #    print(config['opti_method'],end=';',flush=True)
    
    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    data_real = config["data_real"]
    points_real = normalize_points(data_real[0], real_img)
    if config["my"]:
        def f(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]

            (p,v), proj = trackFeatures(proj, data_real, config), proj
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            ret = 0
            for axis, mult in [(0,1),(1,1),(2,1)]:
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    ret += 50
                else:
                    ret += obj
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyRot(cur_x, eps[0], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            (p,v), proj = trackFeatures(projs[:,0], data_real, config), projs[:,0]
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            cur_obj = []
            for axis, mult in [(0,1),(1,1),(2,1)]:
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    cur_obj.append(50)
                else:
                    cur_obj.append(obj)

            ret = [0,0,0]
            for i in range(1, projs.shape[1]):
                proj = projs[:,i]
                (p,v), proj = trackFeatures(proj, data_real, config), proj
                points = normalize_points(p, proj)
                valid = v==1
                points = points[valid]
                part = 0
                axis, mult = [(0,1),(1,1),(2,1)][i-1]
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    obj = 50

                ret[i-1] = (obj-cur_obj[i-1])*eps[i-1]

            return ret
        
        def hessf(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyRot(cur_x, eps[0], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]))
            dvec.append(applyRot(cur_x, eps[0]*2, 0, 0))
            dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, 0, eps[1]*2, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]*2))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            (p,v), proj = trackFeatures(projs[:,0], data_real, config), projs[:,0]
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            cur_obj = []
            for axis, mult in [(0,1),(1,1),(2,1)]:
                obj = calcPointsObjective(axis, points, points_real[valid])*mult
                if obj==-1:
                    cur_obj.append(50)
                else:
                    cur_obj.append(obj)
            
            cur_obj = np.array(cur_obj)

            objs = np.zeros((9,3), dtype=cur_obj.dtype)
            for i in range(1, projs.shape[1]):
                proj = projs[:,i]
                (p,v), proj = trackFeatures(proj, data_real, config), proj
                points = normalize_points(p, proj)
                valid = v==1
                points = points[valid]
                part = 0
                axis = [(0,),(1,),(2,),(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)][i-1]
                for ax in axis:
                    obj = calcPointsObjective(axis, points, points_real[valid])
                    if obj==-1:
                        obj = 50
                    objs[i-1][ax] = obj
                
            ret = np.zeros((3,3), dtype=cur_obj.dtype)
            ret[0,0] = np.sum( (objs[3] - objs[0] - objs[0] + cur_obj) / (eps[0]*eps[0]) )
            ret[0,1] = np.sum( (objs[4] - objs[0] - objs[1] + cur_obj) / (eps[0]*eps[1]) )
            ret[0,2] = np.sum( (objs[5] - objs[0] - objs[2] + cur_obj) / (eps[0]*eps[0]) )
            #ret[1,0] = np.sum( (objs[4] - objs[1] - objs[0] + cur_obj) / (eps[1]*eps[0]) )
            ret[1,0] = ret[0,1]
            ret[1,1] = np.sum( (objs[6] - objs[1] - objs[1] + cur_obj) / (eps[1]*eps[1]) )
            ret[1,2] = np.sum( (objs[7] - objs[1] - objs[2] + cur_obj) / (eps[1]*eps[2]) )
            #ret[2,0] = np.sum( (objs[5] - objs[2] - objs[0] + cur_obj) / (eps[2]*eps[0]) )
            ret[2,0] = ret[0,2]
            #ret[2,1] = np.sum( (objs[7] - objs[2] - objs[1] + cur_obj) / (eps[2]*eps[1]) )
            ret[2,1] = ret[1,2]
            ret[2,2] = np.sum( (objs[8] - objs[2] - objs[2] + cur_obj) / (eps[2]*eps[2]) )
            
            return ret

        cur = correctTrans(cur, config)
        
        eps = [0.5, 0.5, 0.5]
        ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 20, 'eps': eps})

        eps = [0.05, 0.05, 0.05]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 40, 'eps': eps})
        eps = [0.005, 0.005, 0.005]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 40, 'eps': eps})
        eps = [0.0025, 0.0025, 0.0025]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 40, 'eps': eps})

        cur = applyRot(cur, ret.x[0], ret.x[1], ret.x[2])

        #cur = correctTrans(cur, config)
        
        config["angle_noise"] += ret.x
    
    else:
        def f(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            cur_x = applyRot(cur_x, x[3], x[4], x[5])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
            ret = 1-calcGIObjective(real_img, proj, config)
            #print(ret)
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            cur_x = applyRot(cur_x, x[3], x[4], x[5])
            dvec = [cur_x]
            dvec.append(applyTrans(cur_x, eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[2]))
            dvec.append(applyRot(cur_x, eps[3], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[4], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[5]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            h0 = 1-calcGIObjective(real_img, projs[:,0], config)
            #ret = [0,0,0,0,0,0]
            ret = [0,0,0]
            for i in range(1, projs.shape[1]):
                ret[i-1] = (1-calcGIObjective(real_img, projs[:,i], config)-h0) * eps[i-1]
            
            #print(ret)

        def hessf(x, cur, eps):
            cur_x = applyRot(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyRot(cur_x, eps[0], 0, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], 0))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]))
            dvec.append(applyRot(cur_x, eps[0]*2, 0, 0))
            dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], eps[1], 0))
            dvec.append(applyRot(cur_x, 0, eps[1]*2, 0))
            dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            #dvec.append(applyRot(cur_x, eps[0], 0, eps[2]))
            #dvec.append(applyRot(cur_x, 0, eps[1], eps[2]))
            dvec.append(applyRot(cur_x, 0, 0, eps[2]*2))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            h0 = 1-calcGIObjective(real_img, projs[:,0], config)

            objs = np.zeros((9,))
            for i in range(1, projs.shape[1]):
                objs[i-1] = 1-calcGIObjective(real_img, projs[:,i], config)
            
            ret = np.zeros((3,3))
            ret[0,0] = (objs[3] - objs[0] - objs[0] + h0) / (eps[0]*eps[0])
            ret[0,1] = (objs[4] - objs[0] - objs[1] + h0) / (eps[0]*eps[1])
            ret[0,2] = (objs[5] - objs[0] - objs[2] + h0) / (eps[0]*eps[0])
            #ret[1,0] = (objs[4] - objs[1] - objs[0] + h0) / (eps[1]*eps[0])
            ret[1,0] = ret[0,1]
            ret[1,1] = (objs[6] - objs[1] - objs[1] + h0) / (eps[1]*eps[1])
            ret[1,2] = (objs[7] - objs[1] - objs[2] + h0) / (eps[1]*eps[2])
            #ret[2,0] = (objs[5] - objs[2] - objs[0] + h0) / (eps[2]*eps[0])
            ret[2,0] = ret[0,2]
            #ret[2,1] = (objs[7] - objs[2] - objs[1] + h0) / (eps[2]*eps[1])
            ret[2,1] = ret[1,2]
            ret[2,2] = (objs[8] - objs[2] - objs[2] + h0) / (eps[2]*eps[2])
            
            return ret



    cur_x = np.array([0,0,0])
    for its,eps in [(20,0.5), (30,0.1), (30,0.01)]:
        cur = correctTrans(cur, config)
        if config['opti_method'] in ("BFGS","SLSQP", "L-BFGS-B", "TNC"):
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,[eps,eps,eps]),
                                    method=config['opti_method'],
                                    jac=gradf,
                                    options={'maxiter': its, 'eps': [eps,eps,eps]})
        elif config['opti_method'] in ("COBYLA",):
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,[eps,eps,eps]),
                                    method=config['opti_method'],
                                    options={'maxiter': its, 'rhobeg': eps})
        else:
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,[eps,eps,eps]),
                                    method=config['opti_method'],
                                    jac=gradf,
                                    hess=hessf,
                                    options={'maxiter': its})
        cur_x = np.array(ret.x)
        cur = applyRot(cur, cur_x[0], cur_x[1], cur_x[2])

        eps = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
        ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=(cur,eps), method='L-BFGS-B', jac=gradf, options={'maxiter': 20, 'eps': eps})
        eps = [0.05, 0.05, 0.05, 0.005, 0.005, 0.005]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B', jac=gradf, options={'maxiter': 20, 'eps': eps})
        eps = [0.025, 0.025, 0.025, 0.0025, 0.0025, 0.0025]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B', jac=gradf, options={'maxiter': 20, 'eps': eps})
        
        cur = applyTrans(cur, ret.x[0], ret.x[1], ret.x[2])
        cur = applyRot(cur, ret.x[3], ret.x[4], ret.x[5])

        config["trans_noise"] += ret.x[:3]
        config["angle_noise"] += ret.x[3:]

    reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return cur


def bfgs_trans(cur, reg_config, c):
    config = dict(default_config)
    config.update(reg_config)
    config["my"] = c==1

    real_img = config["real_img"]
    noise = config["noise"]
    Ax = config["Ax"]

    if "data_real" not in config or config["data_real"] is None:
        config["data_real"] = findInitialFeatures(real_img, config)
    
    if noise is None:
        config["noise"] = np.zeros((2,3))
        noise = config["noise"]

    trans_noise, angles_noise = noise
    config["angle_noise"] = np.array(angles_noise)
    config["trans_noise"] = np.array(trans_noise)

    data_real = config["data_real"]
    points_real = normalize_points(data_real[0], real_img)
    if config["my"]:
        def f(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]

            (p,v), proj = trackFeatures(proj, data_real, config), proj
            points = normalize_points(p, proj)
            valid = v==1
            points = points[valid]
            ret = 0
            for axis, mult in [(-1,1),(-2,1),(-3,3)]:
                obj = calcPointsObjective(axis, points, points_real[valid])
                if obj==-1:
                    ret += 50*mult
                else:
                    ret += obj*mult
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyTrans(cur_x, eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, -eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 2*eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 2*-eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, -eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 2*eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 2*-eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[2]))
            dvec.append(applyTrans(cur_x, 0, 0, -eps[2]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*eps[2]))
            dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[2]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            ret = [0,0,0]
            for j in range(3):
                i = j*4+1
                def calc_obj(i):
                    proj = projs[:,i]
                    (p,v), proj = trackFeatures(proj, data_real, config), proj
                    points = normalize_points(p, proj)
                    valid = v==1
                    points = points[valid]
                    axis, mult = [(-1,1),(-2,1),(-3,1)][j]
                    obj = calcPointsObjective(axis, points, points_real[valid])*mult
                    if obj==-1:
                        obj = 50
                    return obj
                
                h1 = calc_obj(i)
                h_1 = calc_obj(i+1)
                h2 = calc_obj(i+2)
                h_2 = calc_obj(i+3)
                ret[j] = (-h2+8*h1-8*h_1+h_2)/12

            return ret

        #cur = correctXY(cur, config)
        #cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
        
        eps = [0.5, 0.5, 5]
        ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 50, 'eps': eps})

        eps = [0.25, 0.25, 0.5]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 60, 'eps': eps})
        eps = [0.05, 0.05, 0.25]
        ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                      jac=gradf,
                                      options={'maxiter': 70, 'eps': eps})

        cur = applyTrans(cur, ret.x[0], ret.x[1], ret.x[2])

        #cur = correctXY(cur, config)
        #cur = correctZ(cur, config)
        #cur = correctXY(cur, config)
        
        config["trans_noise"] += ret.x
    
    else:
        
        config["GIoldold"] = None

        def f(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            proj = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
            ret = calcGIObjective(real_img, proj, config)
            #print(ret)
            return ret
        
        def gradf(x, cur, eps):
            cur_x = applyTrans(cur, x[0], x[1], x[2])
            dvec = [cur_x]
            dvec.append(applyTrans(cur_x, eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, -eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*-eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, -eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*-eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, -eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[2]))
            dvec = np.array(dvec)
            projs = Projection_Preprocessing(Ax(dvec))

            #import SimpleITK as sitk
            #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/jac.nrrd")
            #exit()

            h0 = calcGIObjective(real_img, projs[:,0], config)
            ret = [0,0,0]
            for i in range(1, projs.shape[1]):
            #for j in range(3):
            #    i = j*4+1
                ret[i-1] = (calcGIObjective(real_img, projs[:,i], config)-h0) * 0.5
            #    h1 = (calcGIObjective(real_img, projs[:,i], config))
            #    h_1 = (calcGIObjective(real_img, projs[:,i+1], config))
            #    h2 = (calcGIObjective(real_img, projs[:,i+2], config))
            #    h_2 = (calcGIObjective(real_img, projs[:,i+3], config))
            #    ret[j] = (-h2+8*h1-8*h_1+h_2)/12
            
            #print(ret)

            return ret


        if c==2:
            eps = [1, 1, 5]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.5, 0.5, 1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
            
            eps = [0.1, 0.1, 0.5]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
        elif c==-2:
            eps = [1, 1, 5]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 150, 'eps': eps})

            eps = [0.5, 0.5, 1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 150, 'eps': eps})
        elif c==-3:
            eps = [0.5, 0.5, 2]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='BFGS',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
        elif c==-4:
            eps = [0.5, 0.5, 2]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 10, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 10, 'eps': eps})
        elif c==-5:
            eps = [0.5, 0.5, 2]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.25, 0.25, 0.5]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

    cur = applyTrans(cur, ret.x[0], ret.x[1], ret.x[2])

    config["trans_noise"] += ret.x[:3]

    reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return cur


def bfgs_trans_all(curs, reg_config, c):
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

    def calc_obj(cur_proj, i, k):
        (p,v) = trackFeatures(cur_proj, config["data_real"][i], config)
        points = normalize_points(p, cur_proj)
        valid = v==1
        points = points[valid]
        axis, mult = [(-1,1),(-2,1),(-3,1)][k]
        obj = calcPointsObjective(axis, points, config["points_real"][i][valid])*mult
        if obj<0:
            obj = 50
        return obj

    def f(x, curs, eps):
        perftime = time.perf_counter() # 100 s / 50 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x.append(applyTrans(cur, x[pos], x[pos+1], x[pos+2]))
        cur_x = np.array(cur_x)
        proj = Projection_Preprocessing(Ax(cur_x))
        
        q = queue.Queue()
        def t_obj(q,indices):
            np.seterr("raise")
            if config["my"]:
                for i in indices:
                    q.put(calc_obj(proj[:,i], i, i%3))
            else:
                for i in indices:
                    q.put(calcGIObjective(real_img[i], proj[:,i], i, cur_x[i], config))
        
        for u in np.array_split(list(range(len(curs))), 8):
            t = threading.Thread(target=t_obj, args = (q, u))
            t.daemon = True
            t.start()
        for _ in range(proj.shape[1]):
            ret += q.get()
        
        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret/len(curs)

    def f_(x, curs, eps):
        perftime = time.perf_counter() # 185.5 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x.append(applyTrans(cur, x[pos], x[pos+1], x[pos+2]))
        proj = Projection_Preprocessing(Ax(np.array(cur_x)))
        if config["my"]:
            for i in range(len(curs)):
                ret += calc_obj(proj[:,i], i, i%3)
        else:
            for i in range(len(curs)):
                ret += calcGIObjective(real_img[i], proj[:,i], i, cur_x[i], config)
        
        print("obj_", time.perf_counter()-perftime)
        return ret

    def gradf(x, curs, eps):
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
        
        q = queue.Queue()
        def t_obj(q,indices):
            np.seterr("raise")
            if config["my"]:
                for j in indices:
                    pos = j*4
                    h0 = calc_obj(projs[:,pos], j, 0), calc_obj(projs[:,pos], j, 1), calc_obj(projs[:,pos], j, 2)
                    for i in range(3):
                        q.put((j*3+i, (calc_obj(projs[:,pos+i+1], j, i)-h0[i])*0.5))
            else:
                for j in indices:
                    pos = j*4
                    h0 = calcGIObjective(real_img[j], projs[:,pos], j, dvec[pos], config)
                    for i in range(3):
                        q.put((j*3+i, (calcGIObjective(real_img[j], projs[:,pos+i+1], j, dvec[pos+i+1], config)-h0) * 0.5))

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
    
    def gradf3(x, curs, eps):
        perftime = time.perf_counter() # 150 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
            #dvec.append(cur_x)
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
        
        q = queue.Queue()
        def t_obj(q,indices):
            np.seterr("raise")
            if config["my"]:
                for j in indices:
                    pos = j*12
                    for i in range(3):
                        h1 = calc_obj(projs[:,pos+i*4], j, i)
                        h_1 = calc_obj(projs[:,pos+i*4+1], j, i)
                        h2 = calc_obj(projs[:,pos+i*4+2], j, i)
                        h_2 = calc_obj(projs[:,pos+i*4+3], j, i)
                        q.put((j*3+i, (-h2+8*h1-8*h_1+h_2)/12))
            else:
                for j in indices:
                    pos = j*12
                    for i in range(3):
                        h1 = (calcGIObjective(real_img[j], projs[:,pos+i*4], j, dvec[pos+i], config))
                        h_1 = (calcGIObjective(real_img[j], projs[:,pos+i*4+1], j, dvec[pos+i+1], config))
                        h2 = (calcGIObjective(real_img[j], projs[:,pos+i*4+2], j, dvec[pos+i+2], config))
                        h_2 = (calcGIObjective(real_img[j], projs[:,pos+i*4+3], j, dvec[pos+i+3], config))
                        q.put((j*3+i, (-h2+8*h1-8*h_1+h_2)/12))

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

    def gradf_(x, curs, eps):
        perftime = time.perf_counter() # 525 s
        dvec = []
        for i, cur in enumerate(curs):
            pos = i*3
            cur_x = applyTrans(cur, x[pos], x[pos+1], x[pos+2])
            dvec.append(cur_x)
            dvec.append(applyTrans(cur_x, eps[pos], 0, 0))
            #dvec.append(applyTrans(cur_x, -eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*eps[0], 0, 0))
            #dvec.append(applyTrans(cur_x, 2*-eps[0], 0, 0))
            dvec.append(applyTrans(cur_x, 0, eps[pos+1], 0))
            #dvec.append(applyTrans(cur_x, 0, -eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*eps[1], 0))
            #dvec.append(applyTrans(cur_x, 0, 2*-eps[1], 0))
            dvec.append(applyTrans(cur_x, 0, 0, eps[pos+2]))
            #dvec.append(applyTrans(cur_x, 0, 0, -eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*eps[2]))
            #dvec.append(applyTrans(cur_x, 0, 0, 2*-eps[2]))
        dvec = np.array(dvec)
        projs = Projection_Preprocessing(Ax(dvec))

        #import SimpleITK as sitk
        #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/jac.nrrd")
        #exit()

        ret = [0,0,0]*len(curs)
        for j in range(len(curs)):
            pos = j*4
            h0 = calcGIObjective(real_img[j], projs[:,pos], j, dvec[pos], config)
            for i in range(3):
                ret[j*3+i] = (calcGIObjective(real_img[j], projs[:,pos+i+1], j, dvec[pos+i+1], config)-h0) * 0.5
            #    h1 = (calcGIObjective(real_img, projs[:,i], config))
            #    h_1 = (calcGIObjective(real_img, projs[:,i+1], config))
            #    h2 = (calcGIObjective(real_img, projs[:,i+2], config))
            #    h_2 = (calcGIObjective(real_img, projs[:,i+3], config))
            #    ret[j] = (-h2+8*h1-8*h_1+h_2)/12
            
            #print(ret)

        print("grad", time.perf_counter()-perftime)
        return ret


    if config["my"]:
        #if "data_real" not in config or config["data_real"] is None:
        data_real = []
        for img in real_img:
            data_real.append(findInitialFeatures(img, config))
        #config["data_real"] = np.array(real_data)
        config["data_real"] = data_real
        config["points_real"] = [normalize_points(data_real[i][0], real_img[i]) for i in range(len(data_real))]

        if c==-34:
            eps = [0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})

            eps = [0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
            eps = [0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
        elif c==-37:
            eps = [0.5, 0.5, 5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0] * len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})

            eps = [0.25, 0.25, 0.5] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})
            eps = [0.05, 0.05, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp':True})

    else:
        
        config["GIoldold"] = [None]*len(curs)
        config["absp1"] = [None]*len(curs)
        config["p1"] = [None]*len(curs)
        gis = [{} for _ in range(len(curs))]


        if c==2:
            eps = [1, 1, 5]
            ret = scipy.optimize.minimize(f, np.array([0,0,0]), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.5, 0.5, 1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
            
            eps = [0.1, 0.1, 0.5]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})

            eps = [0.05, 0.05, 0.1]
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(cur,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps})
        elif c==-22:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-23:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-24:
            eps = [1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

        elif c==-25:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-26:
            eps = [2, 2, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

            eps = [0.5, 0.5, 1] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 50, 'eps': eps, 'disp': True})

        elif c==-27:
            eps = [1, 1, 10] * len(curs)
            ret = scipy.optimize.minimize(f, np.array([0,0,0]*len(curs)), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.25, 0.25, 2] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})

            eps = [0.01, 0.01, 0.25] * len(curs)
            ret = scipy.optimize.minimize(f, np.array(ret.x), args=(curs,eps), method='L-BFGS-B',
                                        jac=gradf3,
                                        options={'maxiter': 100, 'eps': eps, 'disp': True})


        else:
            print("no method selected", c)
    
    res = []
    for i, cur in enumerate(curs):
        res.append(applyTrans(cur, ret.x[i*3+0], ret.x[i*3+1], ret.x[i*3+2]))

    #config["trans_noise"] += ret.x[:3]
    trans_noise += ret.x.reshape(trans_noise.shape)

    #reg_config["noise"] = (config["trans_noise"], config["angle_noise"])

    return np.array(res), (trans_noise, angles_noise)


def bfgs_all(curs, reg_config, c):
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

    def calc_obj(cur_proj, i, k):
        (p,v) = trackFeatures(cur_proj, config["data_real"][i], config)
        valid = v==1
        p = p[valid]
        points = normalize_points(p, cur_proj)
        if "comps" in config:
            axis, mult = config["comps"][k]
        else:
            axis, mult = [(0,1),(1,1),(2,1),(-1,1),(-2,1),(-3,1)][k]
        obj = calcPointsObjective(axis, points, config["points_real"][i][valid])*mult
        if obj<0:
            obj = 50
        return obj

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
        
        q = queue.Queue()
        def t_obj(q,indices):
            np.seterr("raise")
            if config["my"]:
                for i in indices:
                    q.put(calc_obj(proj[:,i], i, i%6))
            else:
                for i in indices:
                    q.put(calcGIObjective(real_img[i], proj[:,i], i, cur_x[i], config))
        
        for u in np.array_split(list(range(len(curs))), 8):
            t = threading.Thread(target=t_obj, args = (q, u))
            t.daemon = True
            t.start()
        for _ in range(proj.shape[1]):
            ret += q.get()
        
        #print("obj", time.perf_counter()-perftime, ret/len(curs))
        return ret/len(curs)

    def f_(x, curs, eps):
        perftime = time.perf_counter() # 185.5 s
        ret = 0
        cur_x = []
        for i, cur in enumerate(curs):
            pos = i*6
            cur_rot = applyRot(cur, x[pos], x[pos+1], x[pos+2])
            cur_x.append(applyTrans(cur_rot, x[pos+3], x[pos+4], x[pos+5]))
        proj = Projection_Preprocessing(Ax(np.array(cur_x)))
        if config["my"]:
            for i in range(len(curs)):
                ret += calc_obj(proj[:,i], i, i%6)
        else:
            for i in range(len(curs)):
                ret += calcGIObjective(real_img[i], proj[:,i], i, cur_x[i], config)
        
        print("obj_", time.perf_counter()-perftime)
        return ret

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
        
        q = queue.Queue()
        def t_obj(q,indices):
            np.seterr("raise")
            if config["my"]:
                for j in indices:
                    pos = j*7
                    for i in range(6):
                        h0 = calc_obj(projs[:,pos], j, i)
                        q.put((j*3+i, (calc_obj(projs[:,pos+i+1], j, i)-h0)*0.5))
            else:
                for j in indices:
                    pos = j*7
                    h0 = calcGIObjective(real_img[j], projs[:,pos], j, dvec[pos], config)
                    for i in range(3):
                        q.put((j*3+i, (calcGIObjective(real_img[j], projs[:,pos+i+1], j, dvec[pos+i+1], config)-h0) * 0.5))

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
        
        q = queue.Queue()
        def t_obj(q,indices):
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

    def gradf_(x, curs, eps):
        perftime = time.perf_counter() # 525 s
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

        #import SimpleITK as sitk
        #sitk.WriteImage(sitk.GetImageFromArray(projs), "recos/jac.nrrd")
        #exit()

        ret = [0,0,0,0,0,0]*len(curs)
        if config["my"]:
            for j in range(len(curs)):
                pos = j*7
                for i in range(6):
                    h0 = calc_obj(projs[:,pos], j, i)
                    ret[j*6+i] = (calc_obj(projs[:,pos+i+1], j, i)-h0)* 0.5
        else:
            for j in range(len(curs)):
                pos = j*7
                h0 = calcGIObjective(real_img[j], projs[:,pos], j, dvec[pos], config)
                for i in range(6):
                    ret[j*6+i] = (calcGIObjective(real_img[j], projs[:,pos+i+1], j, dvec[pos+i+1], config)-h0) * 0.5

        print("grad", time.perf_counter()-perftime)
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
            config["comps"] = [(10,1),(11,1),(2,1),(-1,1),(-2,1),(-3,1)]
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
            config["comps"] = [(0,1),(1,1),(12,1),(-1,1),(-2,1),(-3,1)]
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
