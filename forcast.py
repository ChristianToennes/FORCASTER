import numpy as np
#import clahe
import cv2
import utils
import matplotlib.pyplot as plt
#import astra
#import SimpleITK as sitk
#import time
import scipy.optimize

def bfgs(index, proj, params, Ax, cali, eps):
    feat_thres=cali['feat_thres']
    real_img = proj[index]
    
    real_img = Projection_Preprocessing(real_img)
    #sitk.WriteImage(sitk.GetImageFromArray(real_img), "./recos/forcast_input.nrrd")
    
    # params for ShiTomasi corner detection
    feature_params = {'feat_thres': feat_thres}

    data_real = findInitialFeatures(real_img, feature_params)

    #eps = np.array([10,10,10,.5,.5])
    cur = np.array(params[index])
    
    def f(x, cur):
        cur_x = applyRot(cur, x[3], x[4], x[5])
        cur_x[0] = x[0:3]
        proj_d = Projection_Preprocessing(Ax( np.array([cur_x]) ) ) [:,0]
        try:
            ret = calcObjective(data_real, real_img, proj_d, feature_params)
            #ret = calcObjectiveStd(data_real, real_img, proj_d, 2, feature_params)
            return ret
        except Exception as e:
            print(e)
            print(x)
            raise e
        
    def gradf(x):
        proj_d = Projection_Preprocessing(Ax( np.array([x]) ) )[:,0]
        ret = estimateJac(np.array([0,0,1]), x, eps, Ax, data_real, real_img, proj_d, feature_params)
        return ret

    proj_d = Projection_Preprocessing(Ax( np.array([cur]) ) )[:,0]
    #cur = roughRegistration(cur, real_img, proj_d, feature_params, Ax, data_real=data_real)
    #print("rough reg f", f(cur))
    ret = scipy.optimize.minimize(f, np.array([0,0,0,0,0,0]), args=cur, method='BFGS', options={'maxiter': 30})
    cur = applyRot(cur, ret.x[3], ret.x[4], ret.x[5])
    cur[0] = ret.x[0:3]
    #print(ret)
    #eps = np.array([5,5,5,.25,.25])
    #ret = scipy.optimize.minimize(f, ret.x, method='L-BFGS-B', options={'eps':eps})
    #print(ret)
    #eps = np.array([1,1,1,.25,.25])
    #ret = scipy.optimize.minimize(f, ret.x, method='L-BFGS-B', options={'eps':eps})
    #print(ret)
    proj_d = Projection_Preprocessing(Ax(np.array([cur])))[:,0]
    return cur, ret.fun, np.sum((real_img-proj_d)**2)

def correctXY(in_cur, vec, points_real, points_new):
    cur = np.array(in_cur)
    #print("xy")
    if len(points_new.shape) == 1:
        diff = np.array([[n.pt[0]-r.pt[0], n.pt[1]-r.pt[1]]  for n,r in zip(points_new,points_real)])
        #print(diff)
    else:
        diff = np.array([[n[0]-r[0], n[1]-r[1]]  for n,r in zip(points_new,points_real)])
    if len(diff)>1:
        xdir = vec[6:9]#/np.linalg.norm(vec[6:9])
        ydir = vec[9:12]#/np.linalg.norm(vec[9:12])
        #cur[0] += np.median(diff, axis=0)[0] / np.linalg.norm(vec[6:9])
        #cur[1] += np.median(diff, axis=0)[1] / np.linalg.norm(vec[9:12])
        cur[0] += np.median(diff, axis=0)[0] * xdir# / np.linalg.norm(vec[6:9])
        cur[0] += np.median(diff, axis=0)[1] * ydir# / np.linalg.norm(vec[9:12])
    return cur

def correctZ(in_cur, vec, points_real, points_new):
    cur = np.array(in_cur)
    #print("z")
    #print(points_new.shape, points_real.shape)
    if len(points_new.shape)==1:
        dist_new = np.array([ np.sqrt((n.pt[0]-r.pt[0])**2 + (n.pt[1]-r.pt[1])**2) for n in points_new for r in points_new])
        dist_real = np.array([ np.sqrt((n.pt[0]-r.pt[0])**2 + (n.pt[1]-r.pt[1])**2) for n in points_real for r in points_real])
    else:
        dist_new = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in points_new for r in points_new])
        dist_real = np.array([ np.sqrt((n[0]-r[0])**2 + (n[1]-r[1])**2) for n in points_real for r in points_real])
    #dist_new[np.bitwise_and(dist_real==0,dist_new==0)] = 1
    if np.count_nonzero(dist_new) > 5:
        #scale = 1-np.median(dist_real[dist_new!=0]/dist_new[dist_new!=0])
        scale = np.median(dist_real[dist_new!=0]/dist_new[dist_new!=0])-1
        #print(scale, len(dist_new[dist_new!=0]), len(dist_real[dist_new!=0]))
        zdir = vec[3:6]/np.linalg.norm(vec[3:6])
        #cur[2] += scale * np.linalg.norm(vec[0:3])
        cur[0] += scale * zdir #* np.linalg.norm(vec[3:6])
    return cur

def correctRot(in_cur, points_real, points_new, Ax, real_img, data_real, eps=0.01, change=None):
    cur = np.array(in_cur)
    dcur = calcJacVectors(cur, np.array([0,0,0,eps,eps,eps]))
    dcur.append(cur)
    dcur = np.array(dcur)
    #print(dcur)
    proj_d = Projection_Preprocessing(Ax(dcur))
    GIoldold = GI(real_img, real_img)
    diff = [calcObjective(data_real, real_img, proj_d[:,proj], {}, GIoldold) for proj in range(proj_d.shape[1])]

    hess1 = (diff[1]-2*diff[-1]+diff[0]) / (eps**2)
    hess2 = (diff[3]-2*diff[-1]+diff[2]) / (eps**2)
    hess3 = (diff[5]-2*diff[-1]+diff[4]) / (eps**2)
    jac1 = (-1.5*diff[1]+2*diff[-1]-0.5*diff[0]) / eps
    jac2 = (-1.5*diff[3]+2*diff[-1]-0.5*diff[2]) / eps
    jac3 = (-1.5*diff[5]+2*diff[-1]-0.5*diff[4]) / eps

    if hess1!=0:
        up1 = jac1/hess1
    if hess2!=0:
        up2 = jac2/hess2
    if hess3!=0:
        up3 = jac3/hess3
    
    old_cur = np.array(cur)

    if up1 != 0:
        #print(diff[0], jac1, diff[0]/jac1)
        cur = applyRot(cur, up1, 0, 0)
    if up2 != 0:
        #print(diff[1], jac2, diff[1]/jac2)
        cur = applyRot(cur, 0, up2, 0)
    if up3 != 0:
        #print(diff[1], jac2, diff[1]/jac2)
        cur = applyRot(cur, 0, 0, up3)
    
    #print(diff[-1]/jac1,diff[-1]/jac2,diff[-1]/jac3)
    #print(diff)
    #print(jac1, jac2, jac3)
    #print(old_cur-cur)
    if change is not None:
        change.clear()
        change.append(up1)
        change.append(up2)
        change.append(up3)
    return cur


def correctRot_(in_cur, points_real, points_new, Ax, real_img, data_real, eps=0.5, comp=0, change=None):
    #print(len(points_new), len(points_real))
    cur = np.array(in_cur)
    if len(points_new) > 1:
        #diff = np.array([calcObjectiveStdPoints(comp+i, points_new, points_real) for i in range(3)])
        #dcur = np.array(calcJacVectors(cur, np.array([0,0,0,eps,eps,eps])))
        
        if comp == 0:
            cur_p = applyRot(cur, eps, 0, 0)
            cur_m = applyRot(cur, -eps, 0, 0)
        if comp == 1:
            cur_p = applyRot(cur, 0, eps, 0)
            cur_m = applyRot(cur, 0, -eps, 0)
        if comp == 2:
            cur_p = applyRot(cur, 0, 0, eps)
            cur_m = applyRot(cur, 0, 0, -eps)
        dcur = np.array([cur_p, cur_m])
        #print(dcur)
        proj_d = Projection_Preprocessing(Ax(dcur))
        #sitk.WriteImage(sitk.GetImageFromArray(proj_d), "recos/jac.nrrd")
        #exit(0)
        #plt.figure()
        #plt.imshow(proj_d[:,2])
        #plt.figure()
        #plt.imshow(proj_d[:,3])
        #plt.show()
        #plt.close()

        points=[]
        valid=[]
        if len(points_real.shape)==1:
            for (p,v), proj in [(trackFeatures_(real_img, proj_d[:,i], data_real, {}), proj_d[:,i]) for i in range(proj_d.shape[1])]:
                points.append(normalize_points(p, proj))
                valid.append(v==1)
        else:
            for (p,v),proj in [(trackFeatures(real_img, proj_d[:,i], data_real, {}), proj_d[:,i]) for i in range(proj_d.shape[1])]:
                points.append(normalize_points(p, proj))
                valid.append(v==1)
        combined_valid = valid[0]
        for v in valid:
            combined_valid = np.bitwise_and(combined_valid, v)
        points = np.array([p[combined_valid] for p in points])
        valid = np.array(valid)

        points_new = normalize_points(points_new, proj_d[:,0])
        points_real = normalize_points(points_real, real_img)

        points_r = np.array([normalize_points(data_real[0][combined_valid], proj_d[:,0]) for v in valid ])
        #print(points.shape, np.count_nonzero(valid, axis=1))

        hess = lambda x_m, x, x_p, eps: x_m-2*x+x_p / (eps**2)
        #hess = lambda x_m, x, x_p, eps: 1
        jac = lambda x_m, x, x_p, eps: -1.5*x_m+2*x-0.5*x_p / eps

        #print(calcJacVectors(cur, np.array([0,0,0,eps,eps,eps]))[1]-cur)
        #print(cur)
        #print(cur-calcJacVectors(cur, np.array([0,0,0,eps,eps,eps]))[0])
        #diffs = [calcObjectiveStdPoints(comp+i//2, points[i][valid[i]], data_real[0][valid[i]] ) for i in range(len(points))]
        #diffs.append(calcObjectiveStdPoints(comp+0, points_new, points_real))
        #diffs.append(calcObjectiveStdPoints(comp+1, points_new, points_real))
        #diffs.append(calcObjectiveStdPoints(comp+2, points_new, points_real))
        #print(diffs[1]-diffs[-3], diffs[0]-diffs[-3], diffs[3]-diffs[-2], diffs[2]-diffs[-2], diffs[5]-diffs[-1], diffs[4]-diffs[-1])
        #print([(len(points[i]), len(points_r[i])) for i in range(len(points))])
        #print([calcObjectiveStdPoints(0, points[i], points_r[i] ) for i in range(len(points))])
        #print([calcObjectiveStdPoints(1, points[i], points_r[i] ) for i in range(len(points))])
        #print([calcObjectiveStdPoints(2, points[i], points_r[i] ) for i in range(len(points))])
        #d = [(calcObjectiveStdPoints(0, points[1], points_r[1] ), calcObjectiveStdPoints(0, points_new, points_real), calcObjectiveStdPoints(0, points[0], points_r[0] )),
        #(calcObjectiveStdPoints(0, points[3], points_r[3] ), calcObjectiveStdPoints(0, points_new, points_real), calcObjectiveStdPoints(0, points[2], points_r[2] )),
        #(calcObjectiveStdPoints(0, points[5], points_r[5] ), calcObjectiveStdPoints(0, points_new, points_real), calcObjectiveStdPoints(0, points[4], points_r[4] )),
        #(calcObjectiveStdPoints(1, points[1], points_r[1] ), calcObjectiveStdPoints(1, points_new, points_real), calcObjectiveStdPoints(1, points[0], points_r[0] )),
        #(calcObjectiveStdPoints(1, points[3], points_r[3] ), calcObjectiveStdPoints(1, points_new, points_real), calcObjectiveStdPoints(1, points[2], points_r[2] )),
        #(calcObjectiveStdPoints(1, points[5], points_r[5] ), calcObjectiveStdPoints(1, points_new, points_real), calcObjectiveStdPoints(1, points[4], points_r[4] )),
        #(calcObjectiveStdPoints(2, points[1], points_r[1] ), calcObjectiveStdPoints(2, points_new, points_real), calcObjectiveStdPoints(2, points[0], points_r[0] )),
        #(calcObjectiveStdPoints(2, points[3], points_r[3] ), calcObjectiveStdPoints(2, points_new, points_real), calcObjectiveStdPoints(2, points[2], points_r[2] )),
        #(calcObjectiveStdPoints(2, points[5], points_r[5] ), calcObjectiveStdPoints(2, points_new, points_real), calcObjectiveStdPoints(2, points[4], points_r[4] ))]
        #print(d[0:3])
        #print(d[3:6])
        #print(d[6:9])
        #print([(jac(x_m,x,x_p,eps), hess(x_m,x,x_p,eps), jac(x_m,x,x_p,eps)/hess(x_m,x,x_p,eps)) for (x_m,x,x_p) in d[0:3]])
        #print([(jac(x_m,x,x_p,eps), hess(x_m,x,x_p,eps), jac(x_m,x,x_p,eps)/hess(x_m,x,x_p,eps)) for (x_m,x,x_p) in d[3:6]])
        #print([(jac(x_m,x,x_p,eps), hess(x_m,x,x_p,eps), jac(x_m,x,x_p,eps)/hess(x_m,x,x_p,eps)) for (x_m,x,x_p) in d[6:9]])
        #exit(0)
        if change is not None:
            change.clear()

        if np.count_nonzero(valid[0])!=0 and np.count_nonzero(valid[1])!=0:
            diff0 = calcObjectiveStdPoints(comp, points_new, points_real)
            diff1 = calcObjectiveStdPoints(comp, points[0], points_r[0] )
            diff2 = calcObjectiveStdPoints(comp, points[1], points_r[1] )
            if diff0 < diff1 and diff0 < diff2:
                hess1 = 0
                jac1 = 0
                #if change is not None:
                #    change.append(1)
            else:
                hess1 = hess(diff2, diff0, diff1, eps)
                jac1 = jac(diff2, diff0, diff1, eps)
            if jac1!=0:
                up11 = diff0/jac1
            else:
                up11 = 0
            if hess1!=0:
                up1 = jac1/hess1
            else:
                up1 = 0
        else:
            print("no jac1")
            #jac1 = 0
            #hess1=0
            up1 = 0
            up11 = 0

        #print(jac1, hess1, calcObjectiveStdPoints(comp+0, points_new, points_real), up1)
        if np.abs(up1) < 1:
            if comp==0:
                #print(cur[3], jac1/hess1, diff[0]*jac1/hess1, diff[0])
                #cur[3] -= diff[0]/jac1
                cur = applyRot(cur, up1, 0, 0)
            if comp==1:
                #print(diff[1], jac2, diff[1]/jac2)
                #cur[4] -= diff[1]/jac2
                cur = applyRot(cur, 0, up1, 0)
            if comp==2:
                #print(diff[1], jac2, diff[1]/jac2)
                #cur[5] -= diff[2]/jac3
                cur = applyRot(cur, 0, 0, up1)
        else:
            up1 = 0
        
        #print(diff[0]/jac1,diff[1]/jac2,diff[2]/jac3)
        #print(diff)
        #print(jac1, jac2, jac3)
        #print(cur, up1, up11, up2, up22, up3, up33)
        if change is not None:
            change.append(up1)
            #change.append(up2)
            #change.append(up3)
    return cur

opt_opts = {"L-BFGS-B": {'gtol':1e-16,'ftol':1e-16,'maxiter':20},
            "BFGS": {'gtol':1e-8,'maxiter':20,"eps":0.001},
            "SLSQP":{"ftol":1e-8,"maxiter":20,"eps":0.001},
            "trust-krylov":{"gtol":1e-8}
            }

def correctR(in_cur, Ax, data_real, real_img, eps, feature_params={}, method='L-BFGS-B', α1=1, α2=10):
    #GIoldold = GI(real_img,real_img)
    cur = np.array(in_cur)
    def f(x):
        cur_x = np.array(cur)
        cur_x[3:] = x
        proj_d = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
        try:
            ret = calcObjective(data_real, real_img, proj_d, feature_params, GIoldold=GIoldold)
            return ret
        except Exception as e:
            print(e)
            print(x)
            raise e
    def f1(x):
        cur_x = applyRot(cur, x, 0, 0)
        proj_d = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
        try:
            ret = calcObjectiveStd(data_real, real_img, proj_d, 0, feature_params)
            return ret
            ret1 = calcObjective(data_real, real_img, proj_d, feature_params, GIoldold=GIoldold)
            #print(1, ret, ret1, α1*ret + α2*ret1)
            return α1*ret + α2*ret1
        except Exception as e:
            print(e)
            print(x)
            raise e
    def f2(x):
        cur_x = applyRot(cur, 0, x, 0)
        proj_d = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
        try:
            ret = calcObjectiveStd(data_real, real_img, proj_d, 1, feature_params)
            return ret
            ret1 = calcObjective(data_real, real_img, proj_d, feature_params, GIoldold=GIoldold)
            #print(2, ret, ret1)
            return α1*ret + α2 * ret1
        except Exception as e:
            print(e)
            print(x)
            raise e
    def f3(x):
        cur_x = applyRot(cur, 0, 0, x)
        proj_d = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
        try:
            ret = calcObjectiveStd(data_real, real_img, proj_d, 2, feature_params)
            return ret
            ret1 = calcObjective(data_real, real_img, proj_d, feature_params, GIoldold=GIoldold)
            #print(2, ret, ret1)
            return α1*ret + α2 * ret1
        except Exception as e:
            print(e)
            print(x)
            raise e
    def gradf(x):
        cur_x = np.array(cur)
        cur_x[3:] = x
        proj_d = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
        ret = estimateJac(cur_x, np.array([0,0,0,eps,eps,eps]), Ax, data_real, real_img, proj_d, feature_params)
        return ret
    
    def hessf(x):
        cur_x = np.array(cur)
        cur_x[3:] = x
        proj_d = Projection_Preprocessing(Ax(np.array([cur_x])))[:,0]
        ret = estimateHessian(cur_x, np.array([0,0,0,eps,eps,eps]), Ax, data_real, real_img, proj_d, feature_params)[1]
        return ret

    #perftime = time.perf_counter()
    #method=None
    #options = dict()
    #options['eps'] = eps
    #ret = scipy.optimize.minimize(f, np.array([0,0]), method=method, options=options)
    #print(method, "none", eps, ret)
    #print(time.perf_counter()-perftime)
    
    #perftime = time.perf_counter()
    #method='L-BFGS-B'
    #options = dict(opt_opts[method])
    #options['eps'] = eps
    #ret = scipy.optimize.minimize(f, np.array([0,0]), method=method, options=options)
    #cur[3:] += ret.x
    if False:
        method='L-BFGS-B'
        options = dict(opt_opts[method])
        #options['eps'] = eps
        #options['maxiter'] = 5
        ret = scipy.optimize.minimize(f1, np.array([0]), method=method, options=options)
        #print(ret)
        cur = applyRot(cur, ret.x, 0, 0)
    if False:
        method='L-BFGS-B'
        options = dict(opt_opts[method])
        #options['eps'] = eps
        #options['maxiter'] = 5
        ret = scipy.optimize.minimize(f2, np.array([0]), method=method, options=options)
        #print(ret)
        cur = applyRot(cur, 0, ret.x, 0)
    if False:
        method='L-BFGS-B'
        options = dict(opt_opts[method])
        #options['eps'] = eps
        #options['maxiter'] = 5
        ret = scipy.optimize.minimize(f3, np.array([0]), method=method, options=options)
        #print(ret)
        cur = applyRot(cur, 0, 0, ret.x)
    if True:
        method='BFGS'
        options = dict(opt_opts[method])
        options['eps'] = eps
        #options['maxiter'] = 5
        ret = scipy.optimize.minimize(lambda x: f1(x[0])+f2(x[1]), np.array([0, 0]), method=method, options=options)
        #print(ret)
        cur = applyRot(cur, ret.x[0], ret.x[1], 0)
    #print(method, "none", eps, ret)
    #print(time.perf_counter()-perftime)
    
    #perftime = time.perf_counter()
    #method='L-BFGS-B'
    #options = dict(opt_opts[method])
    #options['eps'] = eps
    #ret = scipy.optimize.minimize(f, np.array([0,0]), jac='2-point', method=method, options=options)
    #print(method, "2-point", eps, ret)
    #print(time.perf_counter()-perftime)
    #perftime = time.perf_counter()
    #method='L-BFGS-B'
    #options = dict(opt_opts[method])
    #options['eps'] = eps
    #ret = scipy.optimize.minimize(f, np.array([0,0]), jac='3-point', method=method, options=options)
    #print(method, "3-point", eps, ret)
    #print(time.perf_counter()-perftime)
    #perftime = time.perf_counter()
    #method='L-BFGS-B'
    #options = dict(opt_opts[method])
    #options['eps'] = eps
    #ret = scipy.optimize.minimize(f, np.array([0,0]), jac=gradf, method=method, options=options)
    #print(method, "my grad", eps, ret)
    #print(time.perf_counter()-perftime)
    return cur

def lessRoughRegistration(cur, real_img, proj_img, feature_params, Ax, data_real=None):
    if data_real is None:
        data_real = findInitialFeatures(real_img, feature_params)
    #points, valid = trackFeatures_(real_img, proj_img, data_real, feature_params)
    for i2 in range(0):
        break
        if len(data_real[0].shape)==1:
            points, valid = trackFeatures_(real_img, proj_img, data_real, feature_params)
        else:
            points, valid = trackFeatures(real_img, proj_img, data_real, feature_params)
        cur = correctXY(cur, Ax.create_vecs(np.array([cur]))[0], data_real[0][valid], points[valid])
        cur = correctZ(cur, Ax.create_vecs(np.array([cur]))[0], data_real[0][valid], points[valid])
        proj_img = Projection_Preprocessing(Ax(np.array([cur])))[:,0]
    for i1 in range(1):
        #perftime = time.perf_counter()
        if len(data_real[0].shape)==1:
            points, valid = trackFeatures_(real_img, proj_img, data_real, feature_params)
        else:
            points, valid = trackFeatures(real_img, proj_img, data_real, feature_params)
        #cur = correctR(cur, Ax, data_real, real_img, 0.05)
        cur = correctR(cur, Ax, data_real, real_img, 0.01)
        #print(time.perf_counter()-perftime, 10**(-i1))
        #cur = correctXY(cur, vec, data_real[0][valid], points[valid])
        #cur = correctZ(cur, vec, data_real[0][valid], points[valid])
        proj_img = Projection_Preprocessing(Ax(np.array([cur])))[:,0]
        #print(np.count_nonzero(valid), end=',', flush=True)
    #print(cur, flush=True)
    #exit(0)
    for i1 in range(0):
        break
        if len(data_real[0].shape)==1:
            points, valid = trackFeatures_(real_img, proj_img, data_real, feature_params)
        else:
            points, valid = trackFeatures(real_img, proj_img, data_real, feature_params)
        cur = correctXY(cur, Ax.create_vecs(np.array([cur]))[0], data_real[0][valid], points[valid])
        cur = correctZ(cur, Ax.create_vecs(np.array([cur]))[0], data_real[0][valid], points[valid])
        proj_img = Projection_Preprocessing(Ax(np.array([cur])))[:,0]
    return cur

def binsearch(in_cur, data_real, real_img, Ax, axis, my=True, grad_width=(1,5)):
    points_real, features_real = data_real
    points_real = normalize_points(points_real, real_img)
    real_img = Projection_Preprocessing(real_img)
    if not my:
        GIoldold = GI(real_img, real_img)

    make_εs = lambda size, count: np.array([-size/(2**i) for i in range(count)] + [0] + [size/(2**i) for i in range(count)][::-1])
    εs = make_εs(*grad_width)
    change = 0
    selected_εs = []
    cur = np.array(in_cur)
    failed_count = grad_width[0]
    osci_count = 0
    min_ε = 0
    while(True):
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
            points = []
            valid = []
            if len(data_real[0].shape)==1:
                    for (p,v), proj in [(trackFeatures_(real_img, projs[:,i], data_real, {}), projs[:,i]) for i in range(projs.shape[1])]:
                        points.append(normalize_points(p, proj))
                        valid.append(v==1)
            else:
                for (p,v),proj in [(trackFeatures(real_img, projs[:,i], data_real, {}), projs[:,i]) for i in range(projs.shape[1])]:
                    points.append(normalize_points(p, proj))
                    valid.append(v==1)
            combined_valid = valid[0]
            for v in valid:
                combined_valid = np.bitwise_and(combined_valid, v)
            points = [p[v] for p, v in zip(points, valid)]
            
            #print(points.shape, points_real.shape)
            values = np.array([calcObjectiveStdPoints(axis, points, points_real[v]) for points,v in zip(points,valid)])
            #values1 = np.array([calcObjective(data_real, real_img, projs[:,i], {}, GIoldold) for i in range(projs.shape[1])])
            p = np.polyfit(εs, values, 2)
            #p1 = np.polyfit(εs, values1, 2)
        else:
            values = np.array([calcObjective(data_real, real_img, projs[:,i], {}, GIoldold) for i in range(projs.shape[1])])
            p = np.polyfit(εs, values, 2)
        #plt.figure()
        #plt.plot(np.linspace(1.2*εs[0],1.2*εs[-1]), np.polyval(p, np.linspace(1.2*εs[0],1.2*εs[-1])))
        #plt.scatter(εs, values)
        #plt.figure()
        #plt.plot(np.linspace(1.2*εs[0],1.2*εs[-1]), np.polyval(p1, np.linspace(1.2*εs[0],1.2*εs[-1])))
        #plt.scatter(εs, values1)
        #plt.show()
        #plt.close()
        if False:
            min_arg = np.argmin(values)
            min_ε = εs[min_arg]
            change += min_ε
            #print(min_arg, len(εs)//6, len(εs)-len(εs)//6-1, len(εs)//2-len(εs)//6, len(εs)//2+len(εs)//6)
            if min_arg <= len(εs)//6 or min_arg >= len(εs)-len(εs)//6:
                εs *= 2
            elif min_arg >= len(εs)//2-len(εs)//6 and min_arg <= len(εs)//2+len(εs)//6:
                εs *= 0.5
        if True:
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
        if np.abs(min_ε) <= 0.0001:
            break
        if np.abs(change) > failed_count:
            #print("reset", axis, change)
            failed_count += grad_width[0]
            osci_count = 0
            εs = make_εs(*grad_width)
            cur = np.array(in_cur)
            #min_ε = i*np.random.random(1)[0]-0.5*i
            min_ε = εs[np.argmin(values)]
            change = min_ε
        if failed_count > grad_width[0]*5:
            print("failed", end=",")
            break
        selected_εs.append(min_ε)
        if osci_count > 10:
            print("osci", end=",")
            break
        if axis==0:
            cur = applyRot(cur, min_ε, 0, 0)
        if axis==1:
            cur = applyRot(cur, 0, min_ε, 0)
        if axis==2:
            cur = applyRot(cur, 0, 0, min_ε)    
    #print(change)
    return cur


def roughRegistration(cur, real_img, proj_img, feature_params, Ax, data_real=None, c=0, grad_width=(1,5)):
    #print("rough")
    if data_real is None:
        data_real = findInitialFeatures(real_img, feature_params)
    if len(data_real[0].shape)==1:
        points, valid = trackFeatures_(real_img, proj_img, data_real, feature_params)
    #else:
    #    points, valid = trackFeatures(real_img, proj_img, data_real, feature_params)
    
    #plt.figure()
    #plt.imshow(real_img)
    #plt.figure()
    #plt.imshow(proj_img)
    #plt.show()
    #plt.close()

    #cur[5] = correctRot(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1)[5]
    #for _ in range(3):
    #    cur[3:6] = correctRot(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1)[3:6]

    if c==-1:
        for _ in range(1):
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1)[1:]
        for _ in range(5):
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.05)[1:]
        for _ in range(5):
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.01)[1:]
    elif c==-2:
        old_cur = np.array(cur)
        change = []
        cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1, comp=0, change=change)[1:]
        eps = 0.1
        i=0
        old_change = [list(change)]
        while np.max(np.abs(change))>0.001 or np.max(np.abs(change))==0:
            if old_change[-1][0]*change[0]<0:
                i+=1
            #print(np.sum(old_change, axis=0))
            if np.max(np.abs(change)) == 0 or i>5:
                eps = eps*0.5
                i=0
                #print(eps, i)
            if eps<=1e-8:
                break
            old_cur = np.array(cur)
            old_change.append(list(change))
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=eps, comp=0, change=change)[1:]
        old_change.append(list(change))
        print(np.sum(old_change, axis=0))

        old_cur = np.array(cur)
        change = []
        cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1, comp=1, change=change)[1:]
        eps = 0.1
        i=0
        old_change = [list(change)]
        while np.max(np.abs(change))>0.001 or np.max(np.abs(change))==0:
            if old_change[-1][0]*change[0]<0:
                i+=1
            #print(np.sum(old_change, axis=0))
            if np.max(np.abs(change)) == 0 or i>5:
                eps = eps*0.5
                i=0
                #print(eps, i)
            if eps<=1e-8:
                break
            old_cur = np.array(cur)
            old_change.append(list(change))
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=eps, comp=1, change=change)[1:]
        old_change.append(list(change))
        print(np.sum(old_change, axis=0))

        old_cur = np.array(cur)
        change = []
        cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1, comp=0, change=change)[1:]
        eps = 0.1
        i=0
        old_change = [list(change)]
        while np.max(np.abs(change))>0.0001 or np.max(np.abs(change))==0:
            if old_change[-1][0]*change[0]<0:
                i+=1
            #print(np.sum(old_change, axis=0))
            if np.max(np.abs(change)) == 0 or i>5:
                eps = eps*0.5
                i=0
                #print(eps, i)
            if eps<=1e-8:
                break
            old_cur = np.array(cur)
            old_change.append(list(change))
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=eps, comp=0, change=change)[1:]
        old_change.append(list(change))
        print(np.sum(old_change, axis=0))

        old_cur = np.array(cur)
        change = []
        cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=0.1, comp=1, change=change)[1:]
        eps = 0.1
        i=0
        old_change = [list(change)]
        while np.max(np.abs(change))>0.0001 or np.max(np.abs(change))==0:
            if old_change[-1][0]*change[0]<0:
                i+=1
            #print(np.sum(old_change, axis=0))
            if np.max(np.abs(change)) == 0 or i>5:
                eps = eps*0.5
                i=0
                #print(eps, i)
            if eps<=1e-8:
                break
            old_cur = np.array(cur)
            old_change.append(list(change))
            cur[1:] = correctRot_(cur, data_real[0][valid], points[valid], Ax, real_img, data_real, eps=eps, comp=1, change=change)[1:]
        old_change.append(list(change))
        print(np.sum(old_change, axis=0))
    elif c==0:
        cur = binsearch(cur, data_real, real_img, Ax, 2, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 0, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 1, grad_width=grad_width)
    elif c==1:
        cur = binsearch(cur, data_real, real_img, Ax, 2, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 0, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 1, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 2, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 0, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 1, grad_width=grad_width)
    elif c==2:
        cur = binsearch(cur, data_real, real_img, Ax, 0, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 1, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 2, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 0, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 1, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 2, grad_width=grad_width)
    elif c==3:
        cur = binsearch(cur, data_real, real_img, Ax, 2, False, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 0, False, grad_width=grad_width)
        cur = binsearch(cur, data_real, real_img, Ax, 1, False, grad_width=grad_width)
    #print(scale)

    #print(len(data_real[0]), np.count_nonzero(valid))
    #cur = correctXY(cur, Ax.create_vecs(np.array([cur]))[0], data_real[0][valid], points[valid])
    #cur = correctZ(cur, Ax.create_vecs(np.array([cur]))[0], data_real[0][valid], points[valid])
    #if c==0:
    #    print(cur)
    
    #minX, minY, maxX, maxY = getBoundingBox(data_real[0][valid])
    #print(minX, minY, maxX, maxY)
    #dX, dY = maxX-minX, maxY-minY
    #minX, minY, maxX, maxY = getBoundingBox(points[valid])
    #print(minX, minY, maxX, maxY)
    #scale = [dX/(maxX-minX), dY/(maxY-minY)]
    #scale = (1-np.mean(scale))# * np.linalg.norm(vec[0:3])
    #print(scale, dX/(maxX-minX), dY/(maxY-minY), (maxX-minX)/dX, (maxY-minY)/dY, np.linalg.norm(vec[0:3]))
    #cur[2] = (1-np.min(scale)) * np.linalg.norm(vec[0:3])
    #print(cur)
    return cur

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

def calcJacVectors(cur, eps):
    curs = []
    for i in range(len(eps)):
        if eps[i] == 0: continue
        cur_p = np.array(cur)
        cur_m = np.array(cur)
        if i < 3:
            cur_p[0, i] += eps[i]
            cur_m[0, i] -= eps[i]
        if i == 3:
            cur_p = applyRot(cur, eps[i], 0, 0)
            cur_m = applyRot(cur, -eps[i], 0, 0)
        if i == 4:
            cur_p = applyRot(cur, 0, eps[i], 0)
            cur_m = applyRot(cur, 0, -eps[i], 0)
        if i == 5:
            cur_p = applyRot(cur, 0, 0, eps[i])
            cur_m = applyRot(cur, 0, 0, -eps[i])
        curs.append(cur_p)
        curs.append(cur_m)
    return curs

def calcHessianVectors(cur, eps):
    curs = calcJacVectors(cur, eps)
    new_curs = calcJacVectors(cur, eps)
    for p_cur in curs:
        new_curs += calcJacVectors(p_cur, eps)
    new_curs = np.array(new_curs)
    return new_curs

def estimateHessian(cur, eps, Ax, data_real, real_img, ini_DRR, feature_params):
    #perftime_jac = time.perf_counter()
    dcur = calcHessianVectors(cur, eps)

    #ccur = np.array([cur]*len(dcur))
    #proj_d = Ax(ccur)
    #proj_d = Projection_Preprocessing(proj_d)
    #sitk.WriteImage(sitk.GetImageFromArray(proj_d), ".\\recos\\undiff_sino.nrrd")
    
    proj_d = Projection_Preprocessing(Ax(dcur))
    #sitk.WriteImage(sitk.GetImageFromArray(proj_d), ".\\recos\\diff_sino.nrrd")

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
    
    #print("jac+hess calculation", time.perf_counter()-perftime_jac)
    return jac, hess

def estimateJac(cur, eps, Ax, data_real, real_img, ini_DRR, feature_params):
    #perftime_jac = time.perf_counter()
    dcur = np.array(calcJacVectors(cur, eps))

    proj_d = Projection_Preprocessing(Ax(dcur))
    #sitk.WriteImage(sitk.GetImageFromArray(proj_d), ".\\recos\\diff_sino.nrrd")
    jac = np.zeros(eps.shape[0])
    f0 = calcObjective(data_real, real_img, ini_DRR, feature_params)
    for proj_index in range(eps.shape[0]):
        f1 = calcObjective(data_real, real_img, proj_d[:,2*proj_index], feature_params)
        f2 = calcObjective(data_real, real_img, proj_d[:,2*proj_index+1], feature_params)
        jac[proj_index] = (-1.5*f0+2*f1-0.5*f2) / eps[proj_index]
    #print("jac calculation", time.perf_counter()-perftime_jac)
    return jac

def GI(old_img, new_img):
    p1 = np.array([(old_img[1:,:-1]-old_img[:-1,:-1]).flatten(),(old_img[:-1,1:]-old_img[:-1,:-1]).flatten()]).T
    p2 = np.array([(new_img[1:,:-1]-new_img[:-1,:-1]).flatten(),(new_img[:-1,1:]-new_img[:-1,:-1]).flatten()]).T
    absp1 = np.linalg.norm(p1, axis=-1)
    absp2 = np.linalg.norm(p2, axis=-1)

    absGrad = absp1*absp2
    gradDot = p1[:,0]*p2[:,0] - p1[:,1]*p2[:,1]
    gradDot[absGrad==0] = 0
    absGrad[absGrad==0] = 1
    w = 0.5*(gradDot / absGrad + 1)
    return np.sum(w*np.min(np.array([absp1, absp2]), axis=0))

def calcObjective(data_old, old_img, new_img, feature_params, GIoldold=None):
    GIoldnew = GI(old_img, new_img)
    if GIoldold is None:
        GIoldold = GI(old_img, old_img)
    return GIoldold / (GIoldnew+1e-8)

def calcObjectiveStd(data_old, old_img, new_img, comp, feature_params):
    points_old, _ = data_old
    if len(points_old.shape)==1:
        points_new, valid = trackFeatures_(old_img, new_img, data_old, feature_params)
    else:
        points_new, valid = trackFeatures(old_img, new_img, data_old, feature_params)
    #points_new, valid = data_new
    #print(points_new.shape, points_old.shape, valid.shape)
    good_new = normalize_points(points_new[valid], new_img)
    good_old = normalize_points(points_old[valid], old_img)
    #print(good_new.shape, good_old.shape)
    
    #dists_new = np.zeros(good_new.shape[0])
    #dists_old = np.zeros(good_old.shape[0])

    if len(good_new) <5:
        return 1000000

    return calcObjectiveStdPoints(comp, good_new, good_old)

def normalize_points(points, img):
    xdim, ydim = img.shape
    if len(points.shape) == 1:
        return np.array([[1000.0*p.pt[0]/xdim, 1000.0*p.pt[1]/ydim] for p in points])
    else:
        ret = np.array(points)
        ret[:,0] *= 1000.0/xdim
        ret[:,1] *= 1000.0/ydim
        return ret

def calcObjectiveStdPoints(comp, good_new, good_old):
    if comp==0:
        d = good_new[:,0]-good_old[:,0]
        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<mean+3*std, d>mean-3*std)]
        #fd = d[np.bitwise_and(d>np.quantile(d,0.1), d<np.quantile(d,0.9))]
        f = np.var(fd)
    elif comp==1:
        d = good_new[:,1]-good_old[:,1]
        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<mean+3*std, d>mean-3*std)]
        #fd = d[np.bitwise_and(d>np.quantile(d,0.1), d<np.quantile(d,0.9))]
        f = np.var( fd )
    elif comp==12:
        #f = np.var( good_new[:,1]-good_old[:,1] )
        mid_n_x, mid_n_y = np.mean(good_new, axis=0)
        mid_o_x, mid_o_y = np.mean(good_old, axis=0)
        ϕ_new = np.arctan2(good_new[:,1]-mid_n_y, good_new[:,0]-mid_n_x)
        ϕ_old = np.arctan2(good_old[:,1]-mid_o_y, good_old[:,0]-mid_o_x)

        ϕ_new[ϕ_new<-np.pi] += 2*np.pi
        ϕ_new[ϕ_new>np.pi] -= 2*np.pi

        ϕ_old[ϕ_old<-np.pi] += 2*np.pi
        ϕ_old[ϕ_old>np.pi] -= 2*np.pi

        d = ϕ_new*180/np.pi-ϕ_old*180/np.pi
        d[d<-180] += 360
        d[d>180] -= 360

        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<mean+3*std, d>mean-3*std)]


        f = np.var( fd )
    elif comp==2:
        d = np.linalg.norm(good_new-good_old, axis=1)

        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<mean+3*std, d>mean-3*std)]

        f = np.var(d)
    elif comp==3:
        f = np.median( good_new[:,0]-good_old[:,0] )
    elif comp==4:
        f = np.median( good_new[:,1]-good_old[:,1] )
    elif comp==5:
        #f = np.var( good_new[:,1]-good_old[:,1] )
        ϕ_new = np.arctan2(good_new[:,1], good_new[:,0])
        ϕ_old = np.arctan2(good_old[:,1], good_old[:,0])
        f = np.median(ϕ_new*180/np.pi-ϕ_old*180/np.pi)
    elif comp==6:
        f = np.mean( good_new[:,0]-good_old[:,0] )
    elif comp==7:
        f = np.mean( good_new[:,1]-good_old[:,1] )
    elif comp==8:
        #f = np.var( good_new[:,1]-good_old[:,1] )
        ϕ_new = np.arctan2(good_new[:,1], good_new[:,0])
        ϕ_old = np.arctan2(good_old[:,1], good_old[:,0])
        f = np.mean(ϕ_new*180/np.pi-ϕ_old*180/np.pi)
    #print(comp, f, len(good_new), int(100*np.count_nonzero(valid)/len(valid)))
    else:
        f = 0
    return f

def calcObjectiveDist(data_old, old_img, new_img, feature_params):
    f1 = 0.0
    its = 1
    for i in range(its):
        points_new, valid = trackFeatures(old_img, new_img, data_old, feature_params)
        points_old, _ = data_old
        #points_new, valid = data_new
        good_new = points_new[valid]
        good_old = points_old[valid]
        
        dists_new = np.zeros(good_new.shape[0])
        dists_old = np.zeros(good_old.shape[0])

        #print(len(good_new))
        if len(good_new) > 0:
            old_p = good_new[-1]
            for i,p in enumerate(good_new):
                if len(good_new.shape)==1:
                    dists_new[i] = np.linalg.norm([(p.pt[0]-old_p.pt[0]),(p.pt[1]-old_p.pt[1])])
                else:
                    dists_new[i] = np.linalg.norm([(p[0]-old_p[0]),(p[1]-old_p[1])])
                old_p = p

            old_p = good_old[-1]
            for i,p in enumerate(good_old):
                if len(good_old.shape)==1:
                    dists_old[i] = np.linalg.norm([(p.pt[0]-old_p.pt[0]),(p.pt[1]-old_p.pt[1])])
                else:
                    dists_old[i] = np.linalg.norm([(p[0]-old_p[0]),(p[1]-old_p[1])])
                old_p = p
        
            f1 += np.sum((dists_new-dists_old)**2)
            #f1 += np.count_nonzero(~valid)*500
            #print(len(good_new),len(valid),f1/len(good_new))
        else:
            #print(len(points_old), len(points_new), len(good_new))
            f1 += len(valid)*500
        #f1 += np.sum(( np.array( [(n.pt[0]-o.pt[0])**2+(n.pt[1]-o.pt[1])**2 for n,o in zip(good_new,good_old)] ) ))
        #f1 += np.sum(( np.array( [ np.linalg.norm([(n.pt[0]-o.pt[0]),(n.pt[1]-o.pt[1])]) for n,o in zip(good_new,good_old)] ) ))
        #f1 += np.count_nonzero(~valid)*20

        #f1 = np.sum(np.linalg.norm( good_new-good_old ), axis=-1))
    return f1 / its

def trackFeatures(base_img, next_img, data, feature_params):
    base_points, f1 = data
    #if len(base_points.shape) == 1:
    #    base_points = np.array([[p.pt[0], p.pt[1]] for p in base_points], dtype=np.float32)
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #print(base_img.shape, next_img.shape)
    p1, st, err = cv2.calcOpticalFlowPyrLK(base_img, next_img, base_points[:,np.newaxis,:], None, **lk_params)
    #print(p1.shape, st.shape)
    return np.squeeze(p1), np.squeeze(st)==1

def trackFeatures_(base_img, next_img, data, feature_params):
    #perftime = time.perf_counter()
    base_points, f1 = data
    new_points, f2 = findInitialFeatures(next_img, feature_params)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
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
    for i,ms in enumerate(matches):
        if len(ms) > 1:
            m,n = ms    
            if True or m.distance < 0.7*n.distance:
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
    gpu_img = None
    detector = None
    mask = None
    gpu_mask = None

    if mask is None:
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[50:-50,50:-50] = True
    
    if use_cpu:
        #detector = cv2.xfeatures2d_SURF.create(100, 4, 3, False, True)
        #detector = cv2.SIFT_create()
        #detector = cv2.AKAZE_create(threshold=0.0001, nOctaves=4, nOctaveLayers=4)
        #detector = cv2.ORB_create(nfeatures=300, scaleFactor=1.4, nlevels=4, edgeThreshold=41, patchSize=41, fastThreshold=5)
        #points, features = detector.detectAndCompute(img, mask)
        #points = np.array(points)
        
        feature_params = {"maxCorners": 200, "qualityLevel": 0.01, "minDistance": 7, "blockSize": 13}
        points = cv2.goodFeaturesToTrack(img, mask=mask, **feature_params)
        points = points[:,0]
        features = None
    else:
        if "feat_thres" in feature_params:
            feat_thres = feature_params["feat_thres"]
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

def Projection_Preprocessing(proj):
    if len(proj.shape) == 2:
        return cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    else:
        return np.swapaxes(np.array([cv2.normalize(proj[:,i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for i in range(proj.shape[1])]), 0,1)
