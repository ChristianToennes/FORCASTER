import numpy as np
import clahe
import cv2
import utils
import matplotlib.pyplot as plt
import astra
import SimpleITK as sitk
import time
import scipy.optimize

eps = 1e-16
def FORCAST(index, proj, priorCT, cali, fullgeo, out_shape, eps, Ax = None):
    #% FORCAST algorithm 
    #% @todo complete it
    
    #%Check whether input is correct and assigne variables
    #print('Checking and assigning parameters')
    #if nargin == 5:
    #    if ~(strcmp(class(varargin{1}), 'det_obj') && strcmp(class(varargin{2}), 'traj_obj') && strcmp(class(varargin{3}), 'img_obj') && strcmp(class(varargin{4}), 'cali_obj'))
    #        print('Wrong input! One or more inputs is/are not the required input')
    #    #end
    #    det     = varargin{1}
    #    traj    = varargin{2}
    #    img     = varargin{3}
    #    cali    = varargin{4}
    #elif nargin == 2:
    #        if ~(strcmp(class(varargin{1}), 'cali_obj'))
    #        print('Wrong input! Input is not a cali_obj')
    #        #end
    #        cali   = varargin{1}
    #        det    = cali.det_obj
    #        traj   = cali.traj_obj
    #        img    = cali.img_obj 
    #else:
    #    print('Wrong number of input. Either set det_obj, traj_obj, img_obj, cali_obj or just cali.obj as input!')
    #end
        
    #print('off','images:initSize:adjustingMag')
    perftime = time.perf_counter()
    gengif=0
    feat_thres=cali['feat_thres']
    iterations=cali['iterations']
    #real_img    = img.raw_data[:,:,proj]
    real_img = proj[index]
    #priorCT     = img.CBCT
    #ini_DRR     = img.drr_data[:,:,proj]
    #ini_pms     = traj.pm_set[:, proj]

    #visu_mode           = cali.visumode
    confidence_thres    = cali['confidence_thres']
    relax_factor        = cali['relax_factor']
    match_thres         = cali['match_thres'] 
    max_ratio           = cali['max_ratio']
    max_distance        = cali['max_distance']
    outlier_confidence  = cali['outlier_confidence']

    up = 1

    detector_shape = [fullgeo['DetectorRowCount'], fullgeo['DetectorColCount']]

    num_feat=np.zeros(iterations)
    norm_rpe=np.zeros(iterations)
    params=np.zeros((iterations,5))
    if Ax is None:
        Ax = utils.Ax_geo_astra(out_shape, priorCT)
    geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], fullgeo['Vectors'][index:index+1])
    ini_DRR = Ax(geo)[:,0]
    ini_DRR = Projection_Preprocessing(ini_DRR)
    
    real_img = Projection_Preprocessing(real_img)
    #sitk.WriteImage(sitk.GetImageFromArray(real_img), "./recos/forcast_input.nrrd")
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7,
                        feat_thres = feat_thres )
    # Parameters for lucas kanade optical flow
    feature_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                        feat_thres = feat_thres )

    data_real = findInitialFeatures(real_img, feature_params)

    vec = geo['Vectors'][0]
    #eps = np.array([1.0,1.0,1.0,.05,.05])
    cur = np.zeros_like(eps)
    jac, hess = estimateHessian(vec, cur, eps, Ax, detector_shape, data_real, real_img, ini_DRR, feature_params)
    i_hess = np.linalg.inv(hess)

    print("initialized", time.perf_counter()-perftime)

    print('Starting Calibration')
    for it_nr in range(iterations):
        print('Iteration ', it_nr)
        perftime = time.perf_counter()

        data0 = trackFeatures(real_img, ini_DRR, data_real, feature_params)

        if len(data0[0]) < 5:
            print("reduce feature threshold")
            feat_thres *= 0.9
            data_real = findInitialFeatures(real_img, feature_params)
            continue

        #print(i_hess)
        #corr = np.linalg.solve(hess, -jac)
        corr = i_hess.dot(-jac)

        step = calcStepSize(vec, corr, cur, eps, Ax, detector_shape, data_real, real_img, ini_DRR, feature_params)
        #print(step, corr)
        s = step*corr
        cur = cur + s
        geo = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], np.array([applyChange(vec, cur)]))
        ini_DRR = Ax(geo)[:,0]
        ini_DRR=Projection_Preprocessing(ini_DRR)

        jac_old = jac
        jac = estimateJac(vec, cur, eps, Ax, detector_shape, data_real, real_img, ini_DRR, feature_params)
        y = jac-jac_old
        sty = s.dot(y)
        yBy = (y@i_hess).dot(y)
        #print(sty, yBy)
        i_hess = i_hess + (sty + yBy)*np.outer(s,s) / (sty*sty) - (np.outer(i_hess.dot(y), s) + (np.outer(s,y) @ i_hess )) / sty
        #print(corr, step)
        #print(jac)
        #print(i_hess)

        num_feat[it_nr] = len(data0[0])

        norm_rpe[it_nr] = calcObjective(data_real, real_img, ini_DRR, feature_params)
        params[it_nr] = cur
        print('loss function {}'.format(norm_rpe[it_nr]))
        print("iteration time", time.perf_counter()-perftime)

        if it_nr%10==0:
            eps *= 0.7
            sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(ini_DRR,0,1)), "./recos/forcast_"+str(it_nr//10)+".nrrd")
        
        if norm_rpe[it_nr] == 0:
            if data0[1].any():
                break
            else:
                feat_thres *= 0.9
                print("reduce feature threshold", feat_thres)
                feature_params['feat_thres'] = feat_thres
                data_real = findInitialFeatures(real_img, feature_params)
    #end
    #cali['cali_pms'][:,proj]   = ini_pms
    #cali['cali_vec'][:,proj]   = vec_dist
    print('Calibrations terminated. {} were found! \n'.format(num_feat[it_nr]))
    #cali.eval_para[proj]= [num_feat,norm_rpe,sum_corr]
    sitk.WriteImage(sitk.GetImageFromArray(np.swapaxes(ini_DRR,0,1)), "./recos/forcast_output.nrrd")

    best = params[np.argmin(norm_rpe)]

    return applyChange(fullgeo['Vectors'][index], best)
    #end

def bfgs(index, proj, priorCT, cali, fullgeo, out_shape, Ax, eps):
    feat_thres=cali['feat_thres']
    real_img = proj[index]
    
    detector_shape = [fullgeo['DetectorRowCount'], fullgeo['DetectorColCount']]

    if Ax is None:
        Ax = utils.Ax_geo_astra(out_shape, priorCT)
    
    real_img = Projection_Preprocessing(real_img)
    #sitk.WriteImage(sitk.GetImageFromArray(real_img), "./recos/forcast_input.nrrd")
    
    # params for ShiTomasi corner detection
    feature_params = {'feat_thres': feat_thres}

    data_real = findInitialFeatures(real_img, feature_params)

    vec = fullgeo['Vectors'][index]
    #eps = np.array([10,10,10,.5,.5])
    cur = np.zeros_like(eps)

    vecs = np.array([vec])
    
    def f(x):
        vecs = np.array([applyChange(vec,x)])
        geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        proj_d = Projection_Preprocessing(Ax(geo_d))[:,0]
        try:
            ret = calcObjective(data_real, real_img, proj_d, feature_params)
            return ret
        except Exception as e:
            print(e)
            print(x)
            raise e
        
    def gradf(x):
        vecs = np.array([applyChange(vec,x)])
        geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        proj_d = Projection_Preprocessing(Ax(geo_d))[:,0]
        ret = estimateJac(vec, x, eps, Ax, detector_shape, data_real, real_img, proj_d, feature_params)
        return ret

    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
    proj_d = Projection_Preprocessing(Ax(geo_d))[:,0]
    cur = roughRegistration(cur, real_img, proj_d, feature_params, vecs[0], data_real=data_real)
    #print("rough reg f", f(cur))
    ret = scipy.optimize.minimize(f, cur, method='L-BFGS-B', options={'eps':eps,'gtol':1e-8,'ftol':1e-8})
    #print(ret)
    #eps = np.array([5,5,5,.25,.25])
    #ret = scipy.optimize.minimize(f, ret.x, method='L-BFGS-B', options={'eps':eps})
    #print(ret)
    #eps = np.array([1,1,1,.25,.25])
    #ret = scipy.optimize.minimize(f, ret.x, method='L-BFGS-B', options={'eps':eps})
    #print(ret)
    vecs = np.array([applyChange(vec, ret.x)])
    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
    proj_d = Projection_Preprocessing(Ax(geo_d))[:,0]
    return applyChange(vec, ret.x), ret.fun, np.sum((real_img-proj_d)**2)

def roughRegistration(cur, real_img, proj_img, feature_params, vec, data_real=None):
    if data_real is None:
        data_real = findInitialFeatures(real_img, feature_params)
    points, valid = trackFeatures(real_img, proj_img, data_real, feature_params)
    diff = np.array([[n.pt[0]-r.pt[0], n.pt[1]-r.pt[1]]  for n,r in zip(points,data_real[0])])
    cur[0] += np.mean(diff[valid], axis=0)[0] / np.linalg.norm(vec[6:9])
    cur[1] += np.mean(diff[valid], axis=0)[1] / np.linalg.norm(vec[9:12])
    minX, minY, maxX, maxY = getBoundingBox(data_real[0][valid])
    #print(minX, minY, maxX, maxY)
    dX, dY = maxX-minX, maxY-minY
    minX, minY, maxX, maxY = getBoundingBox(points[valid])
    #print(minX, minY, maxX, maxY)
    scale = [dX/(maxX-minX), dY/(maxY-minY)]
    #print(scale, dX/(maxX-minX), dY/(maxY-minY), (maxX-minX)/dX, (maxY-minY)/dY, np.linalg.norm(vec[0:3]))
    cur[2] = (1-np.min(scale)) * np.linalg.norm(vec[0:3])
    #print(cur)
    return cur

def getBoundingBox(points):
    minX = points[0].pt[0]
    minY = points[0].pt[1]
    maxX = points[-1].pt[0]
    maxY = points[-1].pt[1]
    for point in points:
        if point.pt[0] < minX:
            minX = point.pt[0]
        if point.pt[1] < minY:
            minY = point.pt[1]
        if point.pt[0] > maxX:
            maxX = point.pt[0]
        if point.pt[1] > maxY:
            maxY > point.pt[1]
    return minX, minY, maxX, maxY

def applyChange(vec, corr):
    vec_new = np.array(vec)
    Rθ = utils.rotMat(corr[3], vec_new[6:9])
    vec_new[0:3] = Rθ.dot(vec_new[0:3])
    vec_new[3:6] = Rθ.dot(vec_new[3:6])
    vec_new[9:12] = Rθ.dot(vec_new[9:12])
    # rot ϕ
    Rϕ = utils.rotMat(corr[4], vec_new[9:12])
    vec_new[0:3] = Rϕ.dot(vec_new[0:3])
    vec_new[3:6] = Rϕ.dot(vec_new[3:6])
    vec_new[6:9] = Rϕ.dot(vec_new[6:9])
    # trans x
    xdir = vec_new[6:9]/np.linalg.norm(vec_new[6:9])
    vec_new[0:3] += corr[0]*xdir
    vec_new[3:6] += corr[0]*xdir
    # trans y
    ydir = vec_new[9:12]/np.linalg.norm(vec_new[9:12])
    vec_new[0:3] += corr[1]*ydir
    vec_new[3:6] += corr[1]*ydir
    # trans z
    zdir = vec_new[0:3]/np.linalg.norm(vec_new[0:3])
    vec_new[0:3] += corr[2]*zdir
    vec_new[3:6] += corr[2]*zdir
    return vec_new

def calcJacVectors(vec, eps):
    vecs = []
    for i, ep in enumerate(eps):
        vec_p = np.array(vec)
        vec_m = np.array(vec)
        if i==0:
            vec_p[0] += ep
            vec_p[3] += ep
            vec_m[0] -= ep
            vec_m[3] -= ep
        elif i==1:
            # trans y
            vec_p[1] += ep
            vec_p[4] += ep
            vec_m[1] -= ep
            vec_m[4] -= ep
        elif i==2:
            # trans z
            vec_p[2] += ep
            vec_p[5] += ep
            vec_m[2] -= ep
            vec_m[5] -= ep
        elif i==3:
            # rot θ
            Rθ = utils.rotMat(ep, vec_p[6:9])
            vec_p[0:3] = Rθ.dot(vec_p[0:3])
            vec_p[3:6] = Rθ.dot(vec_p[3:6])
            #vec_p[6:9] = Rθ.dot(vec_p[6:9])
            vec_p[9:12] = Rθ.dot(vec_p[9:12])
            Rθ = utils.rotMat(-ep, vec_m[6:9])
            vec_m[0:3] = Rθ.dot(vec_m[0:3])
            vec_m[3:6] = Rθ.dot(vec_m[3:6])
            #vec_m[6:9] = Rθ.dot(vec_m[6:9])
            vec_m[9:12] = Rθ.dot(vec_m[9:12])
        elif i==4:
            # rot ϕ
            Rϕ = utils.rotMat(ep, vec_p[9:12])
            vec_p[0:3] = Rϕ.dot(vec_p[0:3])
            vec_p[3:6] = Rϕ.dot(vec_p[3:6])
            vec_p[6:9] = Rϕ.dot(vec_p[6:9])
            #vec_p[9:12] = Rϕ.dot(vec_p[9:12])
            Rϕ = utils.rotMat(-ep, vec_m[9:12])
            vec_m[0:3] = Rϕ.dot(vec_m[0:3])
            vec_m[3:6] = Rϕ.dot(vec_m[3:6])
            vec_m[6:9] = Rϕ.dot(vec_m[6:9])
            #vec_p[9:12] = Rϕ.dot(vec_p[9:12])
        vecs.append(vec_p)
        vecs.append(vec_m)
    return vecs

def calcHessianVectors(vec, eps):
    vecs = calcJacVectors(vec, eps)
    new_vecs = calcJacVectors(vec, eps)
    for p_vec in vecs:
        new_vecs += calcJacVectors(p_vec, eps)
    new_vecs = np.array(new_vecs)
    return new_vecs

def estimateHessian(vec, cur, eps, Ax, detector_shape, data_real, real_img, ini_DRR, feature_params):
    perftime_jac = time.perf_counter()
    dvec = calcHessianVectors(applyChange(vec, cur), eps)

    cvec = np.array([applyChange(vec, cur)]*len(dvec) )
    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], cvec)
    proj_d = Ax(geo_d)
    proj_d = Projection_Preprocessing(proj_d)
    sitk.WriteImage(sitk.GetImageFromArray(proj_d), ".\\recos\\undiff_sino.nrrd")
    
    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], dvec)
    proj_d = Projection_Preprocessing(Ax(geo_d))
    sitk.WriteImage(sitk.GetImageFromArray(proj_d), ".\\recos\\diff_sino.nrrd")

    jac = np.zeros(eps.shape[0])
    hess = np.zeros((eps.shape[0], eps.shape[0]))

    f0 = calcObjective(data_real, real_img, ini_DRR, feature_params)
    #data_real = (data_real[0][data0[1]], data_real[1][data0[1]])
    #diff = np.linalg.norm(p2[data0[1].flatten()==0]-p1, axis=-1)

    for proj_index in range(eps.shape[0]):
        #data_p = trackFeatures(ini_DRR, proj_d[:,2*proj_index], data_real, feature_params)
        #data_m = trackFeatures(ini_DRR, proj_d[:,2*proj_index+1], data_real, feature_params)
        
        f_p = calcObjective(data_real, real_img, proj_d[:,2*proj_index], feature_params)
        f_m = calcObjective(data_real, real_img, proj_d[:,2*proj_index+1], feature_params)

        jac[proj_index] = (f_p-f0) / eps[proj_index]
        hess[proj_index,proj_index] = (f_p+f_m-2*f0) / (eps[proj_index]**2)
        for proj_index2 in range(eps.shape[0]):
            if proj_index == proj_index2: continue
            # calculate optical flow
            h_index = 2*(eps.shape[0] + proj_index*eps.shape[0] + proj_index2)
            #data_p = trackFeatures(ini_DRR, proj_d[:,h_index], data_real, feature_params)
            #data_m = trackFeatures(ini_DRR, proj_d[:,h_index+1], data_real, feature_params)
            
            f_p = calcObjective(data_real, real_img, proj_d[:,h_index], feature_params)
            f_m = calcObjective(data_real, real_img, proj_d[:,h_index+1], feature_params)
            hess[proj_index,proj_index2] = (f_p+f_m-2*f0) / (eps[proj_index]*eps[proj_index2])
    
    print("jac+hess calculation", time.perf_counter()-perftime_jac)
    return jac, hess

def estimateJac(vec, cur, eps, Ax, detector_shape, data_real, real_img, ini_DRR, feature_params):
    #perftime_jac = time.perf_counter()
    dvec = np.array(calcJacVectors(applyChange(vec,cur), eps))

    geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], dvec)
    proj_d = Projection_Preprocessing(Ax(geo_d))
    #sitk.WriteImage(sitk.GetImageFromArray(proj_d), ".\\recos\\diff_sino.nrrd")
    jac = np.zeros(eps.shape[0])
    f0 = calcObjective(data_real, real_img, ini_DRR, feature_params)
    for proj_index in range(eps.shape[0]):
        f_p = calcObjective(data_real, real_img, proj_d[:,2*proj_index], feature_params)
        jac[proj_index] = (f_p-f0) / eps[proj_index]
    #print("jac calculation", time.perf_counter()-perftime_jac)
    return jac

def calcStepSize(vec, d, cur, eps, Ax, detector_shape, data_real, real_img, ini_DRR, feature_params):
    perftime = time.perf_counter()
    c1 = 1e-4
    c2 = 0.9
    α = 0
    t = 1
    β = np.inf

    def f(x):
        vecs = np.array([applyChange(vec,x)])
        geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        proj_d = Projection_Preprocessing(Ax(geo_d))
        ret = calcObjective(data_real, real_img, proj_d[:,0], feature_params)
        return ret

    def ḟ(x):
        vecs = np.array([applyChange(vec,x), applyChange(applyChange(vec,x), d)])
        geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        proj_d = Projection_Preprocessing(Ax(geo_d))
        f0 = calcObjective(data_real, real_img, proj_d[:,0], feature_params)
        f1 = calcObjective(data_real, real_img, proj_d[:,1], feature_params)
        ret = (f1-f0) / np.linalg.norm(d)
        return ret
    
    def gradf(x):
        vecs = np.array([applyChange(vec,x)])
        geo_d = astra.create_proj_geom('cone_vec', detector_shape[0], detector_shape[1], vecs)
        proj_d = Projection_Preprocessing(Ax(geo_d))[:,0]
        ret = estimateJac(vec, x, eps, Ax, detector_shape, data_real, real_img, proj_d, feature_params)
        return ret

    #return scipy.optimize.line_search(f, gradf, cur, d)[0]
    ḟx = ḟ(cur)
    fx = f(cur) 

    if True:
        α = 0.1
        pre_fx = f(cur+α*d)
        for i in range(1000):
            cur_fx = f(cur+1.5*α*d)
            if cur_fx < pre_fx:
                pre_fx = cur_fx
                α = α * 1.5
            else:
                return α
        return α
    if False:
        xa, xb, xc, fa, fb, fc, funcalls = scipy.optimize.bracket(f, cur, cur+0.1*d)
        if fa<fb:
            if fa<fc:
                return xa
            else:
                return xc
        else:
            if fb<fc:
                return xb
            else:
                return xc
            
    while(1):
        x_td = cur+t*d
        #print(α, t, f(x_td) > fx + c1*t*ḟx, ḟ(x_td) < c2*ḟx)
        if f(x_td) > fx + c1*t*ḟx:
            β = t
            t = 0.5*(α+β)
        elif ḟ(x_td) < c2*ḟx:
            α = t
            if β == np.inf:
                t = 2*α
            else:
                t = 0.5*(α+β)
        else:
            if α==0:
                print("line search failed")
                return 0.3
            print("line search", time.perf_counter()-perftime, α)
            return α

def calcObjective(data_old, old_img, new_img, feature_params):
    f1 = 0.0
    its = 1
    for i in range(its):
        points_new, valid = trackFeatures(old_img, new_img, data_old, feature_params)
        points_old, _ = data_old
        #points_new, valid = data_new
        good_new = points_new[valid]
        good_old = points_old[valid]
        
        #f1 += np.sum(( np.array( [[(n.pt[0]-o.pt[0])**2,(n.pt[1]-o.pt[1])**2] for n,o in zip(good_new,good_old)] ) ))
        f1 += np.sum(( np.array( [ np.linalg.norm([(n.pt[0]-o.pt[0]),(n.pt[1]-o.pt[1])]) for n,o in zip(good_new,good_old)] ) ))
        f1 += np.count_nonzero(~valid)*20
        #f1 = np.sum(np.linalg.norm( good_new-good_old ), axis=-1))
    return f1 / its

def trackFeatures(base_img, next_img, data, feature_params):
    #perftime = time.perf_counter()
    base_points, f1 = data
    new_points, f2 = findInitialFeatures(next_img, feature_params)
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    #matcher = cv2.BFMatcher()
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    if f1 is None:
        print("no features in old image")
    if f2 is None:
        #print("no features in new image")
        return base_points, np.zeros(len(base_points), dtype=bool) 


    matches = matcher.knnMatch(f1, f2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = np.zeros((len(matches), 2))
    points = -np.ones(len(base_points), dtype=int)
    valid = np.zeros(len(base_points), dtype=bool)

    # ratio test as per Lowe's paper
    for i,ms in enumerate(matches):
        if len(ms) > 1:
            m,n = ms    
            if m.distance < 0.7*n.distance:
                matchesMask[i,0]=1
                valid[m.queryIdx] = True
                points[m.queryIdx] = m.trainIdx
            else:
                valid[m.queryIdx] = False
        #elif len(ms) == 1:
        #    m = ms[0]
        #    valid[ms.queryIdx] = True
        #    points[m.queryIdx] = m.trainIdx
        

    for i in range(len(base_points)):
        if np.count_nonzero(points==i)>1:
            valid[points==i] = False
            points[points==i] = -1
            for i2,(m,n) in enumerate(matches):
                if m.trainIdx == i:
                    matchesMask[i2,0] = 0

    #img = cv2.drawMatchesKnn(base_img,base_points,next_img,new_points,matches,None,matchesMask=matchesMask)
    #plt.imshow(img)
    #plt.show()
    #plt.close()

    #if np.count_nonzero(valid) == 0:
        #print("tracking points failed")
        #raise Exception("tracking points failed")
        #img = cv2.drawMatchesKnn(base_img,base_points,next_img,new_points,matches,None,matchesMask=matchesMask)
        #plt.imshow(img)
        #plt.show()
        #plt.close()
    return new_points[points], valid

gpu_img = None
detector = None
mask = None
gpu_mask = None
def findInitialFeatures(img, feature_params, use_cpu=True):
    #global detector, gpu_img, mask, gpu_mask
    #perftime = time.perf_counter()
    feat_thres = feature_params["feat_thres"]
    gpu_img = None
    detector = None
    mask = None
    gpu_mask = None

    if mask is None:
        mask = np.zeros_like(img, dtype=int)
        mask[50:-50,50:-50] = True
    
    if use_cpu:
        #detector = cv2.xfeatures2d_SURF.create(feat_thres, 4, 3, False, True)
        #detector = cv2.SIFT_create()
        detector = cv2.AKAZE_create(threshold=0.0001 )
        points, features = detector.detectAndCompute(img, None)
        points = np.array(points)
    else:
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

def Projection_Preprocessing(proj):
    return cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
