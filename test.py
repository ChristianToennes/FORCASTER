import numpy as np
import astra
from numpy.lib.type_check import real
import utils
import time

def dercurve_wls(data, li, curvtype=None, iblock=None, nblock=None):
    yi = data[0]
    wi = data[1]
    #if iblock is not None:
        #ia = iblock:nblock:size(yi,2);
        #yi = yi[:,ia]
        #if len(wi) > 1:
        #    wi = wi(:,ia);

    deriv = wi * (li - yi);

    if len(wi) == 1:
        curv = np.ones_like(deriv)
    else:
        curv = wi
    return deriv, curv

def dercurve_trl(data, li, curvtype = 'oc', iblock=None, nblock=None): 
    yi = data[0]
    bi = data[1]
    ri = data[2]

    #if iblock is not None:
    #    ia = iblock:nblock:size(yi,2);
    #    yi = yi[:,ia]
    #    if len(bi) > 1:
    #        bi = bi[:,ia]
    #    
    #    if len(ri) > 1:
    #        ri = ri[:,ia]    

    #% transmission Poisson likelihood function
    ei = np.exp(-li)
    mi = bi * ei + ri
    deriv = (1 - yi / mi) * (-bi * ei)

    #curv = trl_curvature(yi, bi, ri, li, curvtype)
        
    #% trl_h_dh()
    #% transmission Poisson likelihood function
    #function [h, dh] = trl_h_dh
    h = lambda y,b,r,l: y * np.log(b*np.exp(-l)+r) - (b*np.exp(-l)+r)
    dh = lambda y,b,r,l: (1 - y / (b*np.exp(-l)+r)) * b*np.exp(-l)

    #% Compute optimal surrogate parabola curvatures
    #% for Poisson transmission model based on Erdogan's formula.
    if curvtype=='oc':

        #% compute curvature at l=0
        ni_max = np.zeros_like(yi)
        if np.isscalar(bi): #% scalar bi (must be positive!)
            ni_max = bi * (1 - yi * ri / (bi + ri)**2)
        else:
            i0 = bi > 0;
            if np.isscalar(ri):
                rii = 1;
            else:
                rii = ri[i0];
            ni_max[i0] = bi[i0] * (1 - yi[i0] * rii / (bi[i0] + rii)**2)

        ni_max[ni_max<0] = 0
        ni = ni_max

        if False:
            il0 = li <= 0;
        else: #% trick in C program due to numerical precision issues
            il0 = li < 0.1;

        tmp = h(yi,bi,ri,li) - h(yi,bi,ri,0) - li * dh(yi,bi,ri,li)
        i = ~il0
        tmp[tmp<0] = 0
        ni[i] = 2 / li[i]**2 * tmp[i]

        #if (ni > ni_max).any():
        #%	plot([ni_max(:) ni(:) ni(:)>ni_max(:)])
        #    warning 'large ni'
        #end


    #% Precomputed approximating parabola curvatures
    #% for Poisson transmission model.
    #% The minimum returned curvature will be zero.
    #% This is compatible with trpl/trp_init_der02_sino() in aspire.
    elif curvtype=='pc':

        #% ni = (yi-ri)^2 / yi, if yi > ri >= 0 and bi > 0
        ii = (yi > ri) & (ri >= 0) & (bi > 0) #% good rays
        ni = np.zeros_like(yi)
        ni[ii] = (yi[ii] - ri[ii])**2 / yi[ii]

    #% newton curvatures (current 2nd derivative)
    elif curvtype=='nc':
        bel = bi * np.exp(-li)
        yb = bel + ri
        ni = (1 - ri*yi/yb**2) * bel

    else:
        raise 'unknown curve type: ' + curvtype

    return deriv, ni

def reco(raw_projs, geo, real_image, iters, b, β, p, α = 0.1):

    μ = np.zeros_like(real_image)
    y = np.array(raw_projs)

    Aμ = utils.Ax_astra(real_image.shape, geo)
    Aty = utils.Atb_astra(real_image.shape, geo)

    nsub = 3
    step0 = 0
    precon = 1
    restart = np.inf #% restart every this many iterations (default: never)
    curvtype = 'pc'

    pot_quad = lambda C1x: np.ones_like(C1x)
    def pot_huber(C1x, d = 0.001):
        w = np.ones_like(C1x)
        ii = np.abs(C1x) > d
        w[ii] = d / np.abs(C1x[ii])
        return w

    pot, dercurv_type = p.split('_')
    if pot=="huber": 
        pot = pot_huber
    else:
        pot = pot_quad

    def Rdercurv(C1x):
        deriv = β * pot(C1x) * C1x
        curv = β * pot(C1x)
        return deriv, curv

    if dercurv_type=='wls':
        dercurv = dercurve_wls
    elif dercurv_type == 'tlr':
        dercurv = dercurve_trl
        curvtype = 'oc'
    else:
        dercurv_type = 'wls'
        dercurv = dercurve_wls

    if np.isscalar(b):
        b̂ = b*np.ones_like(raw_projs)
    else:
        b̂ = np.array(b[:])
    r = np.zeros_like(raw_projs)

    #% initialize projections
    ŷ = Aμ(μ)
    def C1(x):
        C1x = np.zeros_like(x)
        C1x[:-1] = x[:-1]-x[1:]
        C1x[:,:-1] += x[:,:-1]-x[:,1:]
        C1x[:,:,:-1] += x[:,:,:-1]-x[:,:,1:]
        C1x /= 3
        return C1x
    C1x = C1(μ)

    oldinprod = 0

    error = []
    obj_func = []
    proctime = time.perf_counter()

    ddir = newinprod = 0

    y_mean = np.mean(y, axis=(0,2))

    for it in range(iters):
        #ŷ = Ax(μ)
        # update registration
        #for p in range(y.shape[1]):
        #    y[:,p]-ŷ[:,p]
        # update gain

        ŷ_mean = np.mean(ŷ, axis=(0,2))
        b̂ = (1-α) * b̂ + α * b̂/(y_mean/ŷ_mean)

        # update image
        
        #% gradient of cost function
        data = (y, b̂, r)
        [hderiv, hcurv] = dercurv(data, ŷ, curvtype)
        [pderiv, pcurv] = Rdercurv(μ)
        grad = Aty(hderiv) + C1(pderiv)

        #% preconditioned gradient
        pregrad = precon * grad

        #% direction
        newinprod = grad.flatten().dot(pregrad.flatten())
        if oldinprod == 0 or it%restart == 0:
            ddir = -pregrad
            gamma = 0
        else:
            #% todo: offer other step-size rules ala hager:06:aso
            gamma = newinprod / oldinprod; #% Fletcher-Reeves
            ddir = -pregrad + gamma * ddir

        oldinprod = newinprod

        #% check if descent direction
        if ddir.flatten().dot(grad.flatten()) > 0:
            #% reset
            ddir = -pregrad
            oldinprod = 0

        #% step size in search direction
        Adir = Aμ(ddir)
        C1dir = C1(ddir); #% caution: can be a big array for 3D problems

        #% multiple steps based on quadratic surrogates
        step = step0
        for i_s in range(nsub):
            if step != 0:
                [hderiv, hcurv] = dercurv(data, ŷ + step * Adir, curvtype)
                [pderiv, pcurv] = Rdercurv(C1x + step * C1dir)
            denom = (Adir**2).flatten().dot(hcurv.flatten()) + (C1dir**2).flatten().dot(pcurv.flatten())
            numer = Adir.flatten().dot(hderiv.flatten()) + C1dir.flatten().dot(pderiv.flatten())
            if denom == 0:
                print('found exact solution???  step=0 now!?')
                step = 0
            else:
                step = step - numer / denom

        #% update
        ŷ = ŷ + step * Adir
        C1x = C1x + step * C1dir
        μ = μ + step * ddir

        if it%20==0:
            yield μ
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        obj_func.append((time.perf_counter()-proctime, 2))


