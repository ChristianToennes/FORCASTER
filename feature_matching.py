import numpy as np
import cv2

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

    matchesMask2 = np.array(matchesMask)

    if len(dists[valid])>0:
        p_new = np.array([[p.pt[0], p.pt[1]] for p in new_points[points]])
        p_old = np.array([[p.pt[0], p.pt[1]] for p in base_points])
        ps = p_new-p_old
        m = np.mean(ps[valid], axis=0)
        std = np.std(ps[valid], axis=0)
        #print((ps-m)[valid], m, std)
        out = np.bitwise_or(ps<m-1*std, ps>m+1*std)
        out = np.bitwise_or(out[:,0], out[:,1])
        matchesMask2[out, 0] = 0
        #print(np.count_nonzero(out[valid]))
        valid[out] = False

    
    if False:
        real_img = config["real_img"]
        #img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
        #    np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=np.zeros_like(matchesMask), matchColor=(0,255,0), singlePointColor=(0,0,255))
        #cv2.imwrite("featurepoints.png", img)
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask, matchColor=(0,255,0), singlePointColor=(0,0,255))
        #cv2.imwrite("knnmath.png", img)
        f, (ax1, ax2) = plt.subplots(1,2, squeeze=True)
        ax1.imshow(img)
        #f.savefig("knnmath.png")
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,
            matchesMask=matchesMask2, matchColor=(0,255,0), singlePointColor=(0,0,255))
        ax2.imshow(img)
        plt.show()
        plt.close()
        #exit()

    if np.count_nonzero(valid) > 100:
        pass
        #drop = np.arange(len(valid))[valid][100:]
        #valid[drop] = False
        #drop = np.random.choice(np.arange(len(valid))[valid], 100)
        #valid = np.zeros_like(valid)
        #valid[drop] = True
    else:
        pass
        #print(np.count_nonzero(valid))

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

def Projection_Preprocessing(proj, alpha=0, beta=255):
    proj = np.array(proj, dtype=np.float32)
    mean = np.mean(proj)
    std = np.std(proj)
    return (proj-mean) / std

    if len(proj.shape) == 2:
        return cv2.normalize(proj, None, alpha, beta, cv2.NORM_MINMAX).astype('uint8')
    else:
        return np.swapaxes(np.array([cv2.normalize(proj[:,i], None, alpha, beta, cv2.NORM_MINMAX).astype('uint8') for i in range(proj.shape[1])]), 0,1)
