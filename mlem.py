import tigre
import numpy as np
import time
import scipy.ndimage

def gen_cliques(N):

    w_ort = 1
    w_dia = 2**(-1/2)

    for k in np.ndindex(*N):
        if k[0] > 0:
            if k[1] > 0:
                if k[2] > 0:
                    yield (k, (k[0]-1, k[1]-1, k[2]-1), w_dia)
                yield (k, (k[0]-1, k[1]-1, k[2]), w_dia)
                if k[2] < (N[2]-1):
                    yield (k, (k[0]-1, k[1]-1, k[2]+1), w_dia)
            
            if k[2] > 0:
                yield (k, (k[0]-1, k[1], k[2]-1), w_dia)
            yield (k, (k[0]-1, k[1], k[2]), w_ort)
            if k[2] < (N[2]-1):
                yield (k, (k[0]-1, k[1], k[2]+1), w_dia)

            if k[1] < (N[1]-1):
                if k[2] > 0:
                    yield (k, (k[0]-1, k[1]+1, k[2]-1), w_dia)
                yield (k, (k[0]-1, k[1]+1, k[2]), w_dia)
                if k[2] < (N[2]-1):
                    yield (k, (k[0]-1, k[1]+1, k[2]+1), w_dia)
        
        if k[1] > 0:
            if k[2] > 0:
                yield (k, (k[0], k[1]-1, k[2]-1), w_dia)
            yield (k, (k[0], k[1]-1, k[2]), w_ort)
            if k[2] < (N[2]-1):
                yield (k, (k[0], k[1]-1, k[2]+1), w_dia)
        
        if k[2] > 0:
            yield (k, (k[0], k[1], k[2]-1), w_ort)
        
        if k[2] < (N[2]-1):
            yield (k, (k[0], k[1], k[2]+1), w_ort)

        if k[1] < (N[1]-1):
            if k[2] > 0:
                yield (k, (k[0], k[1]+1, k[2]-1), w_dia)
            yield (k, (k[0], k[1]+1, k[2]), w_ort)
            if k[2] < (N[2]-1):
                yield (k, (k[0], k[1]+1, k[2]+1), w_dia)


        if k[0] < (N[0]-1):
            if k[1] > 0:
                if k[2] > 0:
                    yield (k, (k[0]+1, k[1]-1, k[2]-1), w_dia)
                yield (k, (k[0]+1, k[1]-1, k[2]), w_dia)
                if k[2] < (N[2]-1):
                    yield (k, (k[0]+1, k[1]-1, k[2]+1), w_dia)
            
            if k[2] > 0:
                yield (k, (k[0]+1, k[1], k[2]-1), w_dia)
            yield (k, (k[0]+1, k[1], k[2]), w_ort)
            if k[2] < (N[2]-1):
                yield (k, (k[0]+1, k[1], k[2]+1), w_dia)

            if k[1] < (N[1]-1):
                if k[2] > 0:
                    yield (k, (k[0]+1, k[1]+1, k[2]-1), w_dia)
                yield (k, (k[0]+1, k[1]+1, k[2]), w_dia)
                if k[2] < (N[2]-1):
                    yield (k, (k[0]+1, k[1]+1, k[2]+1), w_dia)


def R(fs, q=2,λ=0.2):
    w_sum = 0

    k = np.zeros((3,3,3))
    k[:,:,1] = 2**(-1/2)
    k[:,1,:] = 2**(-1/2)
    k[1,:,:] = 2**(-1/2)
    k[:,1,1] = -1
    k[1,:,1] = -1
    k[1,1,:] = -1
    k[1,1,1] = 8+12*2**(-1/2)

    #k = k / np.sum(k)

    #print(k)

    if q == 2:
        w_sum = scipy.ndimage.convolve(fs, k, mode='constant')
    else:
        for k,j,w in gen_cliques(fs.shape):
            w_sum += w*np.abs(fs[k]-fs[j])**(q-1)*np.sign(fs[k]-fs[j])
    return λ**q * q * w_sum

def mlem(proj, geo, angles, iters, initial=None):
    if initial is None:
        initial = tigre.algorithms.fdk(proj,geo,angles)
        #initial = np.zeros((geo.nVoxel), dtype=np.float32 )
    
    fs = initial

    N = initial.shape
    M = proj.shape
    proctime = time.process_time()
    f = tigre.Atb(np.exp(-proj), geo, angles)
    f[f==0] = 0.1
    print("backprojected data", time.process_time()-proctime, "s mean value:", np.mean(f))

    angles2 = np.linspace(0,2*np.pi,dtype=np.float32)
    next_fs = np.zeros_like(fs)

    for i in range(iters):
        proctime = time.process_time()
        print("start iter ", i)
        ps = np.exp(-tigre.Ax(fs,geo,angles,'interpolated'))
        print("projected ps: ", time.process_time()-proctime, "s mean value:", np.mean(ps), np.median(ps), ps.shape)
        proctime = time.process_time()
        x = tigre.Atb(ps, geo, angles)
        print("backprojected ps: ", time.process_time()-proctime, "s mean value:", np.mean(x), np.median(x), x.shape)
        proctime = time.process_time()
        Rfs = np.sum(R(fs))
        print("calucated regularizer: ", time.process_time()-proctime, "s mean value:", np.mean(Rfs), np.median(Rfs), Rfs.shape)
        proctime = time.process_time()
        factor = x / (f + Rfs)
        factor[factor==np.nan] = 0
        factor[factor==np.inf] = 0
        factor[factor==-np.inf] = 0
        print("update factor: ", time.process_time()-proctime, "s mean value:", np.mean(factor), np.median(factor), factor.shape)
        proctime = time.process_time()
        next_fs = fs * factor
        print("iteration finished: ", i, time.process_time()-proctime, "s change:", np.sum(next_fs-fs), np.median(next_fs-fs), next_fs.shape)
        fs = next_fs
    
    return fs

def mlem1(proj, geo, angles, iters, initial=None):
    if initial is None:
        initial = tigre.algorithms.fdk(proj,geo,angles)
        #initial = np.zeros((geo.nVoxel), dtype=np.float32 )
    
    fs = initial

    N = initial.shape
    M = proj.shape
    proctime = time.process_time()
    f = tigre.Atb(np.exp(-proj), geo, angles)
    f[f==0] = 0.1
    print("backprojected data", time.process_time()-proctime, "s mean value:", np.mean(f))

    angles2 = np.linspace(0,2*np.pi,dtype=np.float32)
    next_fs = np.zeros_like(fs)

    for i in range(iters):
        proctime = time.process_time()
        print("start iter ", i)
        next_μ = μ + 
        print("iteration finished: ", i, time.process_time()-proctime, "s change:", np.sum(next_fs-fs), np.median(next_fs-fs), next_fs.shape)
        fs = next_fs
    
    return fs

def mlem2(proj, geo, angles, iters, initial=None):
    if initial is None:
        initial = tigre.algorithms.fdk(proj,geo,angles)
        #initial = np.zeros((geo.nVoxel), dtype=np.float32 )
    
    fs = initial

    N = initial.shape
    M = proj.shape
    proctime = time.process_time()
    f = tigre.Atb(np.exp(-proj), geo, angles)
    f[f==0] = 0.1
    print("backprojected data", time.process_time()-proctime, "s mean value:", np.mean(f))

    angles2 = np.linspace(0,2*np.pi,dtype=np.float32)
    next_fs = np.zeros_like(fs)

    next_fs = 0.5*(tigre.Ax(fs, geo, angles)-proj).T*C.T*(tigre.Ax(fs, geo, angles)-proj)+R(fs)