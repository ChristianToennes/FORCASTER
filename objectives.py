import numpy as np
import itertools
from feature_matching import *

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
    p2 = new_img[1::gi_skip,:-1:gi_skip,:-1:gi_skip]-new_img[:-1:gi_skip,:-1:gi_skip,:-1:gi_skip], new_img[:-1:gi_skip,1::gi_skip,:-1:gi_skip]-new_img[:-1:gi_skip,:-1:gi_skip,:-1:gi_skip], new_img[:-1:gi_skip,:-1:gi_skip,1::gi_skip]-new_img[:-1:gi_skip,:-1:gi_skip,:-1:gi_skip]
    absp2 = np.sqrt(p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2], dtype=np.float32)
    #absp2 = np.linalg.norm(p2, ord=2, axis=0)
    absGrad = absp1[::gi_skip, ::gi_skip,::gi_skip]*absp2
    minabs = np.min(np.array([absp1[::gi_skip, ::gi_skip, ::gi_skip], absp2]), axis=0)
    #del absp2
    gradDot = p1[0][::gi_skip, ::gi_skip, ::gi_skip]*p2[0] + p1[1][::gi_skip, ::gi_skip, ::gi_skip]*p2[1] + p1[2][::gi_skip, ::gi_skip, ::gi_skip]*p2[2]
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

def GI2D(new_img, p1, absp1, gi_skip=1):
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
    if len(old_img_big.shape) == 2:
        return calcGIObjective2D(old_img_big, new_img_big, i, cur, config)
    global gi_mask, gi_shape
    if gi_mask is None or gi_mask.shape != old_img_big.shape:
        gi_mask = np.zeros_like(old_img_big, dtype=bool)
        b1 = old_img_big.shape[0]//5
        b2 = old_img_big.shape[1]//5
        b3 = old_img_big.shape[2]//5
        gi_mask[b1:-b1,b2:-b2,b3:-b3] = True
        gi_shape = (old_img_big.shape[0]-b1-b1, old_img_big.shape[1]-b2-b2, old_img_big.shape[2]-b3-b3)

    old_img = old_img_big[gi_mask].reshape(gi_shape)
    new_img = new_img_big[gi_mask].reshape(gi_shape)
    #if cur is not None:
    #    for key in gis[i].keys():
    #        k=np.array(key)
    #        if np.linalg.norm(k[0]-cur[0]) < 0.01 and np.linalg.norm(k[1]-cur[1]) < 0.01 and np.linalg.norm(k[2]-cur[2]) < 0.01:
    #            return gis[i][key]

    if config["GIoldold"][i] is None:
        p1 = old_img[1:,:-1,:-1]-old_img[:-1,:-1,:-1], old_img[:-1,1:,:-1]-old_img[:-1,:-1,:-1], old_img[:-1,:-1,1:]-old_img[:-1,:-1,:-1]
        config["p1"][i] = p1
        config["absp1"][i] = np.sqrt(p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2],dtype=np.float32)
        #config["absp1"][i] = np.linalg.norm(p1, ord=2, axis=0)
        #config["GIoldold"][i] = np.array([GI(old_img, p1, config["absp1"][i]), GI(old_img, p1, config["absp1"][i], 2), GI(old_img, p1, config["absp1"][i], 4)])
        config["GIoldold"][i] = GI(old_img, p1, config["absp1"][i])
    #perftime = time.perf_counter()
    #GIoldnew =np.array([GI(new_img, config["p1"][i], config["absp1"][i]),GI(new_img, config["p1"][i], config["absp1"][i], 2),GI(new_img, config["p1"][i], config["absp1"][i], 4)])
    GIoldnew =GI(new_img, config["p1"][i], config["absp1"][i])
    #print("GI", time.perf_counter()-perftime)
    #return GIoldnew / config["GIoldold"]
    ngi = np.sum(GIoldnew / config["GIoldold"][i])
    #if cur is not None:
    #    gis[i][tuple((tuple(c) for c in cur))] = ngi
    return 1.0/(ngi+1e-10)

def calcGIObjective2D(old_img_big, new_img_big, i, cur, config):
    global gi_mask, gi_shape
    if gi_mask is None or gi_mask.shape != old_img_big.shape:
        gi_mask = np.zeros_like(old_img_big, dtype=bool)
        b1 = old_img_big.shape[0]//5
        b2 = old_img_big.shape[1]//5
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
        #config["GIoldold"][i] = np.array([GI(old_img, p1, config["absp1"][i]), GI(old_img, p1, config["absp1"][i], 2), GI(old_img, p1, config["absp1"][i], 4)])
        config["GIoldold"][i] = GI2D(old_img, p1, config["absp1"][i])
    #perftime = time.perf_counter()
    #GIoldnew =np.array([GI(new_img, config["p1"][i], config["absp1"][i]),GI(new_img, config["p1"][i], config["absp1"][i], 2),GI(new_img, config["p1"][i], config["absp1"][i], 4)])
    GIoldnew =GI2D(new_img, config["p1"][i], config["absp1"][i])
    #print("GI", time.perf_counter()-perftime)
    #return GIoldnew / config["GIoldold"]
    ngi = np.sum(GIoldnew / config["GIoldold"][i])
    #if cur is not None:
    #    gis[i][tuple((tuple(c) for c in cur))] = ngi
    return 1.0/(ngi+1e-10)

def calcPointsObjective(comp, good_new, good_old, img_shape=(0,0)):
    if len(good_new) == 0:
        return 10000
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
        d = np.sqrt( (good_new[:,0]-good_old[:,0])**2 + (good_new[:,1]-good_old[:,1])**2 )
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

    elif comp==20:
        d_new = (good_new[:,0]-good_new[:,0].T[:,np.newaxis]).flatten()
        d_old = (good_old[:,0]-good_old[:,0].T[:,np.newaxis]).flatten()
        d_new = d_new #/ np.median(np.abs(d_new))
        d_old = d_old #/ np.median(np.abs(d_old))

        d = np.abs(d_new-d_old)
        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]

        f = np.mean(fd)
    
    elif comp==21:
        d_new = (good_new[:,1]-good_new[:,1].T[:,np.newaxis]).flatten()
        d_old = (good_old[:,1]-good_old[:,1].T[:,np.newaxis]).flatten()
        if np.median(np.abs(d_new)) != 0:
            d_new = d_new / np.median(np.abs(d_new))
        if np.median(np.abs(d_old)) != 0:
            d_old = d_old / np.median(np.abs(d_old))

        d = np.abs(d_new-d_old)
        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]

        f = np.mean(fd)
    
    elif comp==30:
        d_new = np.abs((good_new[:,0]-good_new[:,0].T[:,np.newaxis]).flatten())
        d_old = np.abs((good_old[:,0]-good_old[:,0].T[:,np.newaxis]).flatten())
        #d_new = d_new / (np.max(good_new[:,0])-np.min(good_new[:,0]))
        #d_old = d_old / (np.max(good_old[:,0])-np.min(good_old[:,0]))

        #d = d_new-d_old
        #std = np.std(d)
        #mean = np.mean(d)
        #fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]

        #f = np.sum(fd)
        f = np.median(d_new)
    
    elif comp==31:
        d_new = np.abs((good_new[:,1]-good_new[:,1].T[:,np.newaxis]).flatten())
        d_old = np.abs((good_old[:,1]-good_old[:,1].T[:,np.newaxis]).flatten())
        #d_new = d_new / (np.max(good_new[:,1])-np.min(good_new[:,1]))
        #d_old = d_old / (np.max(good_old[:,1])-np.min(good_old[:,1]))

        #d = d_new-d_old
        #std = np.std(d)
        #mean = np.mean(d)
        #fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]

        #f = np.sum(fd)
        f = np.median(d_new)

    elif comp==40:
        mid_new = np.array([img_shape[0]//2, img_shape[1]//2])
        mid_old = np.array([img_shape[0]//2, img_shape[1]//2])
        mid_new = np.array([img_shape[1]//2, img_shape[0]//2])
        mid_old = np.array([img_shape[1]//2, img_shape[0]//2])
        mid_new = np.array([0,0])
        mid_old = np.array([0,0])
        d_new = np.linalg.norm(good_new-mid_new[np.newaxis,:], axis=1)
        d_old = np.linalg.norm(good_old-mid_old[np.newaxis,:], axis=1)
        d_new = d_new / np.median(d_new)
        d_old = d_old / np.median(d_old)
        
        f = np.sum( np.abs( d_new/d_old - 1 ) )

    elif comp==41:
        mid_new = np.array([img_shape[0]//2, img_shape[1]//2])
        mid_old = np.array([img_shape[0]//2, img_shape[1]//2])
        d_new = np.linalg.norm(good_new-mid_new[np.newaxis,:], axis=1)
        d_old = np.linalg.norm(good_old-mid_old[np.newaxis,:], axis=1)
        d_new = d_new / np.median(d_new)
        d_old = d_old / np.median(d_old)
    
        f = np.std( d_new-d_old )

    elif comp==42:
        d_new = np.linalg.norm(good_new, axis=1)
        d_old = np.linalg.norm(good_old, axis=1)
        d_new = d_new / np.median(d_new)
        d_old = d_old / np.median(d_old)
        
        f = np.std( d_new-d_old )

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
    elif comp==-4:
        d = np.linalg.norm(good_new-good_old, ord=2, axis=1)
        std = np.std(d)
        mean = np.mean(d)
        fd = d[np.bitwise_and(d<=mean+3*std, d>=mean-3*std)]

        f = np.median( fd )
    elif comp==-5:
        d = good_new-good_old
        #std = np.std(d, axis=0)
        #mean = np.mean(d, axis=0)
        #filt = np.bitwise_and(d<=mean+2*std, d>=mean-2*std)
        #filt = np.bitwise_or(filt[:,0], filt[:,1])

        d = np.linalg.norm(d, ord=2, axis=1)
        f = np.median( d )
    elif comp==-6:
        d = good_new-good_old
        #std = np.std(d, axis=0)
        #mean = np.mean(d, axis=0)
        #filt = np.bitwise_and(d<=mean+2*std, d>=mean-2*std)
        #filt = np.bitwise_or(filt[:,0], filt[:,1])

        d = np.linalg.norm(d, ord=2, axis=1)
        f = np.mean( d )
    elif comp==-7:
        d = good_new-good_old
        d = np.linalg.norm(d, ord=2, axis=1)
        f = np.sum( d )
    elif comp==-8:
        d = good_new-good_old
        #std = np.std(d, axis=0)
        #mean = np.mean(d, axis=0)
        #filt = np.bitwise_and(d<=mean+2*std, d>=mean-2*std)
        #filt = np.bitwise_or(filt[:,0], filt[:,1])

        d = np.linalg.norm(d, ord=2, axis=1)
        f = np.var( d )
    elif comp==-9:
        #d = good_new-good_old
        #std = np.std(d, axis=0)
        #mean = np.mean(d, axis=0)
        #filt = np.bitwise_and(d<=mean+2*std, d>=mean-2*std)
        #filt = np.bitwise_or(filt[:,0], filt[:,1])
        
        #f = np.count_nonzero(filt)
        f = 1.0 / max(1, good_new.shape[0])
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
