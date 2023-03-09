import numpy as np
import cv2

def trackFeatures(next_img, data, config):
    #perftime = time.perf_counter()

    #base_points, f1 = data
    sim_data = findInitialFeatures(next_img, config)
    # FLANN parameters
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)   # or pass empty dictionary
    #matcher = cv2.FlannBasedMatcher(index_params,search_params)
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return matchFeatures(data, sim_data, config)

def debug_imgs_features(config, next_img, base_points, new_points, matches, matchesMask):
    real_img = config["real_img"]
    i = config["prefix"]
    l=60
    r=-40
    t=25
    b=-25
    #print(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8).shape)
    #img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
    #    np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=np.zeros_like(matchesMask), matchColor=(0,255,0), singlePointColor=(100,100,255))
    #cv2.imwrite(i+"_featurepoints.png", img[t:b,l:r])
    #m = np.zeros_like(matchesMask)
    #m[:,0] = 1
    img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask, matchColor=(0,255,0), singlePointColor=(100,100,255))
    cv2.imwrite(i+"_knnmatch.png", img[t:b,l:r])
    #import matplotlib.pyplot as plt
    #f, (ax1, ax2) = plt.subplots(1,2, squeeze=True)
    #ax1.imshow(img)
    #f.savefig("knnmath.png")
    #img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
    #    np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,
    #    matchesMask=matchesMask2, matchColor=(0,255,0), singlePointColor=(0,0,255))
    #ax2.imshow(img)
    #plt.show()
    #plt.close()

def matchFeatures(real_data, sim_data, config=None, next_img=None):
    base_points, f1 = real_data
    new_points, f2 = sim_data
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    #matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    #matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    if f1 is None:
        print("no features in old image")
    if f2 is None:
        print("no features in new image")
        return base_points, np.zeros(len(base_points), dtype=bool) 

    matches = matcher.knnMatch(f1, f2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = np.zeros((len(matches), 2), dtype=int)
    points = -np.ones(len(base_points), dtype=int)
    valid = np.zeros(len(base_points), dtype=bool)

    # ratio test as per Lowe's paper
    dists = []
    for i,ms in enumerate(matches):
        if len(ms) > 1:
            m,n = ms    
            dists.append(m.distance)
            if m.distance < 0.8*n.distance:
                matchesMask[i,0]=1
                valid[m.queryIdx] = True
                points[m.queryIdx] = m.trainIdx
            else:
                valid[m.queryIdx] = False
        else:
            dists.append(-1)
            valid[ms[0].queryIdx] = False

    dists = np.array(dists)
    #print(np.count_nonzero(valid))
    
    matchesMask2 = np.array(matchesMask)

    for i in range(len(base_points)):
        if np.count_nonzero(points[valid]==i)>1:
            valid[points==i] = False
            points[points==i] = -1
            for i2,(m,n) in enumerate(matches):
                if m.trainIdx == i:
                    matchesMask2[i2,0] = 0

    matchesMask3 = np.array(matchesMask2)
    #print(np.count_nonzero(valid))

    if len(dists[valid])>0:
        p_new = np.array([[p.pt[0], p.pt[1]] for p in new_points[points]])
        p_old = np.array([[p.pt[0], p.pt[1]] for p in base_points])
        ps = p_new-p_old
        m = np.mean(ps[valid], axis=0)
        std = np.std(ps[valid], axis=0)
        #print((ps-m)[valid], m, std)
        out = np.bitwise_or(ps<m-1*std, ps>m+1*std)
        out = np.bitwise_or(out[:,0], out[:,1])
        matchesMask3[out, 0] = 0
        #print(np.count_nonzero(out[valid]))
        valid[out] = False

    #print(np.count_nonzero(valid))
    
    if False:
        real_img = config["real_img"]
        l=60
        r=-40
        t=25
        b=-25
        print(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8).shape)
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=np.zeros_like(matchesMask), matchColor=(0,255,0), singlePointColor=(100,100,255))
        cv2.imwrite("featurepoints.png", img[t:b,l:r])
        m = np.zeros_like(matchesMask)
        m[:,0] = 1
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=m, matchColor=(0,255,0), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask, matchColor=(0,255,0), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe.png", img[t:b,l:r])
        #m[::4,0] = (np.ones_like(matchesMask)-matchesMask)[::4,0]
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=m-matchesMask, matchColor=(0,0,255), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe_diff.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask2, matchColor=(0,255,0), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe_out.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask-matchesMask2, matchColor=(0,0,255), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe_out_diff.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask3, matchColor=(0,255,0), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowet_double.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask2-matchesMask3, matchColor=(0,0,255), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe_double_diff.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask3, matchColor=(0,255,0), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe_out_double.png", img[t:b,l:r])
        img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
            np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,matchesMask=matchesMask-matchesMask3, matchColor=(0,0,255), singlePointColor=(100,100,255))
        cv2.imwrite("knnmatch_lowe_out_double_diff.png", img[t:b,l:r])
        print(np.count_nonzero(m), np.count_nonzero(matchesMask), np.count_nonzero(m-matchesMask))
        print(np.count_nonzero(matchesMask), np.count_nonzero(matchesMask2), np.count_nonzero(matchesMask-matchesMask2))
        print(np.count_nonzero(matchesMask2), np.count_nonzero(matchesMask3), np.count_nonzero(matchesMask2-matchesMask3))
        print(np.count_nonzero(matchesMask), np.count_nonzero(matchesMask3), np.count_nonzero(matchesMask-matchesMask3))
        #import matplotlib.pyplot as plt
        #f, (ax1, ax2) = plt.subplots(1,2, squeeze=True)
        #ax1.imshow(img)
        #f.savefig("knnmath.png")
        #img = cv2.drawMatchesKnn(np.array(255*(real_img-np.min(real_img))/(np.max(real_img)-np.min(real_img)),dtype=np.uint8),base_points,
        #    np.array(255*(next_img-np.min(next_img))/(np.max(next_img)-np.min(next_img)),dtype=np.uint8),new_points,matches,None,
        #    matchesMask=matchesMask2, matchColor=(0,255,0), singlePointColor=(0,0,255))
        #ax2.imshow(img)
        #plt.show()
        #plt.close()
        exit()

    if config is not None and next_img is not None:
        debug_imgs_features(config, next_img, base_points, new_points, matches, matchesMask3)

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
        mask[100:-100,100:-100] = True
    
    if config["use_cpu"]:
        #detector = cv2.xfeatures2d_SURF.create(100, 4, 3, False, True)
        #detector = cv2.SIFT_create(nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6, descriptorType = cv2.CV_8U)
        detector = cv2.AKAZE_create(**config["AKAZE_params"])
        #detector = cv2.xfeatures2d.StarDetector_create(maxSize=45,responseThreshold=10,lineThresholdProjected=10,lineThresholdBinarized=8,suppressNonmaxSize=5)
        #brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        #detector = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=21, fastThreshold=20)
        #img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        #img = np.stack([img, np.zeros_like(img), np.zeros_like(img)], axis=-1)
        #print(img.shape, img.dtype)
        points, features = detector.detectAndCompute(img, np.ones_like(img, dtype=np.uint8))
        #points = detector.detect(img, mask)
        #points, features = brief.compute(np.repeat(img[:,:,np.newaxis], 3, axis=2), points)
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
        return np.array([[100.0*p.pt[0]/xdim, 100.0*p.pt[1]/ydim] for p in points])
    else:
        ret = np.array(points)
        ret[...,0] *= 100.0/xdim
        ret[...,1] *= 100.0/ydim
        return ret

def unnormalize_points(points, img):
    xdim, ydim = img.shape
    ret = np.array(points)
    ret[:,0] *= xdim/100.0
    ret[:,1] *= ydim/100.0
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
