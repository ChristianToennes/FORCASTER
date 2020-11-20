from numpy.lib import utils
import tigre
import astra
import numpy as np
import time
import scipy.ndimage
import utils
import ctypes
import multiprocessing
import functools

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
    k[:,:,1] = -2**(-0.5)
    k[:,1,:] = -2**(-0.5)
    k[1,:,:] = -2**(-0.5)
    k[:,1,1] = -1
    k[1,:,1] = -1
    k[1,1,:] = -1
    k[1,1,1] = 0
    k[1,1,1] = np.sum(np.abs(k))
    #k[1,1,1] = 6+12*2**(-1/2)

    #k = k / np.sum(k)

    #print(k)

    if q == 2:
        w_sum = scipy.ndimage.convolve(fs, k, mode='constant')
    else:
        for k,j,w in gen_cliques(fs.shape):
            w_sum += w*np.abs(fs[k]-fs[j])**(q-1)*np.sign(fs[k]-fs[j])
    return λ**q * q * w_sum


W = np.zeros((3,3,3), dtype=float)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        for k in range(W.shape[2]):
            if i != 1 or j != 1 or k != 1:
                W[i, j, k] = 1.0 / np.sqrt((1-i)*(1-i) + (1-j)*(1-j) + (1-k)*(1-k))

W = W.flatten()

def μm(μ, f):
    return scipy.ndimage.generic_filter(μ, filt, size=3)

δ = 0.001
ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
δδψ = lambda x: 1 if x <= δ else 0

from PotentialFilter import potential_filter as c_ψ
from PotentialFilter import potential_dx_filter as c_δψ
from PotentialFilter import potential_dxdx_filter as c_δδψ
from PotentialFilter import potential_dx_t_filter as c_δψ_t

def c_μm(μ, f, δ=0.0005):
    user_data = ctypes.c_double(δ)
    ptr = ctypes.cast(ctypes.pointer(user_data), ctypes.c_void_p)
    callback = scipy.LowLevelCallable(f(), ptr)
    return scipy.ndimage.generic_filter(μ, callback, size=3)

def filt(data,f):
    return np.sum(w*f(data[13]-d) for d,w in zip(data,W))

def mlem0(proj, geo, angles, iters, initial=None): # aootaphao 2008
    if initial is None:
        initial = tigre.algorithms.fdk(proj,geo,angles)
        #initial = np.ones((geo.nVoxel), dtype=np.float32 )
    
    μshape = initial.shape
    μ = initial.flatten()
    yshape = proj.shape
    b = 100 # i0 values for every pixel
    y = b*np.exp(-proj.flatten())

    β = 0.2 # smootheness factor

    N = len(μ)
    M = len(y)
    proctime = time.process_time()
    f = tigre.Atb(y.reshape(yshape), geo, angles).flatten()
    f[f==0] = 0.1
    print("backprojected data", time.process_time()-proctime, "s mean value:", np.mean(f))

    for i in range(iters):
        proctime = time.process_time()
        print("start iter ", i)
        l = tigre.Ax(μ.reshape(μshape),geo,angles,'interpolated').flatten()
        print("projected lines: ", time.process_time()-proctime, "s mean value:", np.mean(l), np.median(l), l.shape)
        proctime = time.process_time()

        nom = 0.01*tigre.Atb((b*np.exp(-l) - y).reshape(yshape), geo, angles).flatten() - β * μm(μ.reshape(μshape), δψ).flatten()
        print("calculated nominator: ", time.process_time()-proctime, "s mean value:", np.mean(nom), np.median(nom), nom.shape)
        proctime = time.process_time()
        denom = 0.01*tigre.Atb((l*b*np.exp(-l)).reshape(yshape), geo, angles).flatten() + μ*β* μm(μ.reshape(μshape), δδψ).flatten()
        #Rfs = R(μ)
        print("calucated denom: ", time.process_time()-proctime, "s mean value:", np.mean(denom), np.median(denom), denom.shape)
        proctime = time.process_time()
        factor = nom / denom
        print("update factor: ", time.process_time()-proctime, "s mean value:", np.mean(factor), np.median(factor), factor.shape)
        proctime = time.process_time()
        μ = μ +  μ * factor
        μ[μ<0] = 0
        μ[μ==np.nan] = 0
        μ[μ==np.inf] = 0
        print("iteration finished: ", i, time.process_time()-proctime, "s")
    
    return μ.reshape(μshape)

def R1(δ, μ, μt, p):
    return δ

def mlem1(proj, geo, angles, iters, initial=None, p=2): # stayman 2011
    if initial is None:
        #initial = tigre.algorithms.fdk(proj,geo,angles)
        from tigre.demos.Test_data import data_loader
        μt=data_loader.load_head_phantom(number_of_voxels=geo.nVoxel)
        μ = tigre.algorithms.fdk(proj,geo,angles)
        δ = R1(d, μ, μt, p)
        #initial = np.zeros((geo.nVoxel), dtype=np.float32 )
    
    fs = initial

    N = initial.shape
    M = proj.shape
    proctime = time.process_time()
    f = tigre.Atb(np.exp(-proj), geo, angles)
    W = tigre.Atb(np.ones_like(proj), geo, angles)
    f[f==0] = 0.1
    print("backprojected data", time.process_time()-proctime, "s mean value:", np.mean(f))

    angles2 = np.linspace(0,2*np.pi,dtype=np.float32)
    next_fs = np.zeros_like(fs)

    for i in range(iters):
        proctime = time.process_time()
        print("start iter ", i)
        print("update μ")
        μ = μ + SPS(Φ(μ, μt, δ, p))
        print("update δ")
        δ = δ + BFGS(R1(δ, μ, μt, p))
        print("iteration finished: ", i, time.process_time()-proctime, "s change:", np.sum(next_fs-fs), np.median(next_fs-fs), next_fs.shape)
    
    return fs

def R2(μ): # regularizer, input: attenuation values, output: scalar
    return 0

def mlem2(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, use_astra=True): # tilley 2017
    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    
    y = proj
    Nμ = μ.shape # Number of voxels
    Ny = y.shape # Number of measurements
    
    B = b # Gain/blur matrix
    Bt = b
    Bs = np.zeros((*Ny, *Ny)) # spot blur
    Bst = np.zeros((*Ny, *Ny))
    Bd = np.zeros((*Ny, *Ny)) # scintillator blur
    Bdt = np.zeros((*Ny, *Ny))
    G = np.zeros((*Ny, *Ny)) # data scale by bare-beam photon flux
    Gt = np.zeros((*Ny, *Ny))
    K = np.zeros((*Ny, *Ny)) # Measurement covariance
    K = Bd * np.diag(y) * Bdt + np.identity(Ny)*np.std(y)**2
    W = np.zeros((*Ny, *Ny)) #= K.T  Weighting matrix
    BtWB = Gt * Bst * np.diag(1/y) * Bs * G
    R = R2 # regularizer

    # initialization
    η = Bt * W * B * np.ones_like(y)
    γ = Σj_a_ij( np.ones_like(y), geo, angles)
    BtWy = Bt*W*y

    M = [0]

    for n in range(iters):
        proctime = time.process_time()
        #print("start iteration:", p)
        for m in M:
            l = Σj_a_ij(μ, geo, angles)
            x = np.exp(-l)
            ηox = np.convolve(η, x)
            p = Bt*W*B*x - BtWy - ηox
            L = M*Σi_a_ij(( - np.convolve(ηox, x) - np.convolve(p, x) ), geo, angles)
            c = 2*η*p
            lg0 = l>0
            c[lg0] = (2*0.5*η[lg0]+p[lg0]-0.5*η[lg0]*x[lg0]*x[lg0]-x[lg0]*p[lg0]-l[lg0]*(η[lg0]*x[lg0]*x[lg0] + p[lg0]*x[lg0])) / (l[lg0]*l[lg0])
            c[c<0] = 0
            D = M * Σi_a_ij(np.convolve(γ, c), geo, angles)
            Δμ = (L+dφ) / (D + ddφ)
            μ = μ - Δμ
            μ[μ<0] = 0
            if n%100 == 0:
                if real_image is None:
                    print(n, np.mean(Δμ), np.std(Δμ), np.median(Δμ), np.sum(Δμ))
                else:
                    print(n, np.sum(Δμ), np.mean(np.abs(real_image-μ)))


        #print("iteration finished: ", time.process_time()-proctime, "s change:", np.mean(Δμ))
    
    return μ


def p_norm(x, p, δ=1):
    δarea = x < δ
    return np.sum(x[δarea])*math.pow(2*δ, -p)*math.pow(δ*δ*2, p-1) + \
           np.sum( np.abs(x[~δarea] - δ*(1-0.5*p) * np.sign(x[~δarea]) ) ** p ) /p

def PIPLE(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, βp=10**3, use_astra=True): # stayman 2013

    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])
    #λ = 0 # registration parameters (3 angles, 3 translations)
    #H = σI
    r = 0.1

    Ψp = 1 # rigid transformation
    Ψr = 1 # rigid transformation
    βr = 1

    y = b*np.exp(-proj) + r
    N = y.shape

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    c = (y-r)**2 / y
    error = []
    for n in range(iters):
        #for r in range(1, R):
        #    H[r] = BFGS()
        #    Δθ = (ψp*δW(λ)*μp).T * δf(ψp*(μ-W(λ)*μp))
        #    eφ = λ[r-1] + φ * H[r] * ΔΘ(λ[r-1])
        #    λ[r] = λ[r-1] + eφ * H[r] * ΔΘ(λ[r-1])
        #λ[0] = λ[-1]
        #H[0] = H[-1]

        l = Σj_a_ij(μ)
        #ḣ = (y / ( b*np.exp(-l̂) + r ) - 1)*b*np.exp(-l̂)
        ḣ = b*np.exp(-l)-y

        up = (
            Σi_a_ij(ḣ) \
            #- βr * Ψr * δfr * (Ψr*μ)  \
            - βp* Ψp * c_μm(μ, c_δψ)
        ) / (
            Σi_a_ij2( c * l) \
            #+ βr * Ψr**2 * ωfr * (ψr*μ) \
            + βp * Ψp**2 * c_μm(μ, c_δψ_t)
        )
        μ = μ + up
        μ[μ<0] = 0
        error.append(np.mean(np.abs(real_image-μ)))
        if n%1000 == 0:
            if real_image is None:
                print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(n, np.sum(up), np.mean(np.abs(real_image-μ)))
            yield μ[:]
        #print(n, np.mean(μ), np.std(μ), np.median(μ), np.min(μ), np.max(μ))
    yield μ
    yield error

def CCA(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, use_astra=True): # fessler 1995
    y = b*np.exp(-proj)
    
    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])

    r = 0.1
    ω = 0.6

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    error = []
    for i in range(iters):
        l = Σj_a_ij( μ )
        ȳ = b*np.exp(-l) + r
        #L̇ = Σi_a_ij( (1-y/ȳ) * b * np.exp(-l) )
        part = ȳ - r - y + r*y/ȳ
        L̇ = Σi_a_ij( part )

        #L̈ = - Σi_a_ij2( (1-y*r/(ȳ*ȳ)) * b * np.exp(-l) )
        L̈ = -Σi_a_ij2( part + y*r*r/(ȳ*ȳ) )

        Ṗ = c_μm(μ, c_δψ)
        P̈ = c_μm(μ, c_δδψ)

        nom = (L̇ - β * Ṗ)
        den = (-L̈ + β * P̈)

        #print(i, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
        #print(i, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )
        up = ω*nom/den
        μ = μ + up
        μ[μ<0] = 0
        error.append(np.mean(np.abs(real_image-μ)))
        if i%100 == 0:
            if real_image is None:
                print(i, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(i, np.sum(up), np.mean(np.abs(real_image-μ)))
            yield μ[:]
    yield μ
    yield error

def ML_OSTR(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, use_astra=True): # erdogan 1999
    #print("proj", np.mean(proj), np.median(proj), np.max(proj), np.min(proj))
    #print("e^proj", np.mean(np.exp(-proj)), np.median(np.exp(-proj)), np.max(np.exp(-proj)), np.min(np.exp(-proj)))
    
    y = b*np.exp(-proj)+0.1
    #print("y", np.mean(y), np.median(y), np.max(y), np.min(y))
    
    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])

    r = 0.1

    if use_astra:
        c = utils.Ax_astra(out_shape, geo)(np.ones_like(μ))
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)

    l̂ = Σj_a_ij(μ)
    #print("l", np.mean(l̂), np.median(l̂), np.max(l̂), np.min(l̂))

    d = Σi_a_ij( Σj_a_ij(np.ones_like(μ)) * (y-r)**2 / y)
    d_0 = d==0
    #d[d==0] = 0.000001
    M = 1 # subsets
    error = []
    for it in range(iters):
        l̂ = Σj_a_ij(μ)
        ŷ = b*np.exp(-l̂)
        ḣ = (y / ( ŷ + r ) - 1)*ŷ
        n = M * Σi_a_ij(ḣ)
        #print(it, np.mean(n), np.mean(d), np.median(n), np.median(d), np.mean(n/d), np.median(n/d))
        #print(it, np.mean(μ), np.median(μ), np.mean(μ*(n/d)), np.median(μ*(n/d)) )

        up = n/d
        up[d_0]=0
        μ = μ - up
        μ[μ<=0] = 0.000001
        error.append(np.mean(np.abs(real_image-μ)))
        if it%100 == 0:
            if real_image is None:
                print(it, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(it, np.sum(up), np.mean(np.abs(real_image-μ)))
            yield μ[:]
    yield μ
    yield error

def PL_OSTR(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, use_astra=True): # erdogan 1999

    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])

    y = b * np.exp(-proj) + 0.1
    r = 0.1

    M = 1 # subsets

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)

    d = Σi_a_ij( Σj_a_ij(np.ones_like(μ)) * (y-r)**2 / y)
    error = []
    for n in range(iters):
        for m in range(M):
            l̂ = Σj_a_ij(μ)
            ŷ = b*np.exp(-l̂)
            ḣ = (y / ( ŷ + r ))*ŷ
            L̇ = M * Σi_a_ij(ḣ)

            nom = (L̇ + β * c_μm(μ, c_δψ) )
            den = (d + 2*β* c_μm(μ, c_δψ_t) )
            den[den==0] = 0.00001
            #if iter%10 == 0:
            #print(n, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
            #print(n, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )
            up = nom/den
            μ = μ - up
            μ[μ<0] = 0
            error.append(np.mean(np.abs(real_image-μ)))
            if n%100 == 0:
                if real_image is None:
                    print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
                else:
                    print(n, np.sum(up), np.mean(np.abs(real_image-μ)))
                yield μ[:]

    yield μ
    yield error


def update_step(arg):
    y, μ, β, b, out_shape, geo = arg
    Σj_a_ij = utils.Ax_astra(out_shape, geo)
    Σi_a_ij = utils.Atb_astra(out_shape, geo)
    
    l = Σj_a_ij(μ)
    Ṗ = β*c_μm(μ, c_δψ)
    P̈ = μ*β*c_μm(μ, c_δδψ)
    #print(n, np.mean(Ṗ), np.mean(P̈), np.median(Ṗ), np.median(P̈))

    ŷ = b*np.exp(-l)
    nom = Σi_a_ij(ŷ-y)
    den = Σi_a_ij(l*ŷ)

    den = den + P̈
    f = ((nom - Ṗ)/(den))
    f[den==0] = 0
    #print(n, np.mean(nom), np.mean(den), np.median(nom), np.median(den))
    #print(n, np.mean(μ), np.std(μ), np.median(μ), np.mean(μ*f), np.std(μ*f), np.median(μ*f))
    #den[den==0] = 0.00001
    up = μ*f
    return up

def PL_C(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, use_astra=True): # aootaphao 2008

    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])

    y = b * np.exp(-proj)

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)

    #c = Σj_a_ij(μ)
    #print("Σaij", np.mean(c), np.std(c), np.median(c), np.min(c), np.max(c))
    error = []

    #pool_size = 1
    #subsets = 1
    #pool = multiprocessing.Pool(pool_size)
    #geos = [astra.create_proj_geom('cone_vec', geo["DetectorRowCount"], geo["DetectorColCount"], geo["Vectors"][i::subsets]) for i in range(subsets)]
    proctime = time.perf_counter()
    for n in range(iters):
        #l = Σj_a_ij(μ)
        #ups = pool.imap_unordered(update_step, [(y[i::subsets], μ, β, b, out_shape, geos[i]) for i in range(subsets)] )
        #up = functools.reduce(lambda x,y: x+y, ups) / subsets
        l = Σj_a_ij(μ)
        Ṗ = β*c_μm(μ, c_δψ)
        P̈ = μ*β*c_μm(μ, c_δδψ)
        #print(n, np.mean(Ṗ), np.mean(P̈), np.median(Ṗ), np.median(P̈))

        ŷ = b*np.exp(-l)
        nom = Σi_a_ij(ŷ-y)
        den = Σi_a_ij(l*ŷ)

        den = den + P̈
        f = ((nom - Ṗ)/(den))
        f[den==0] = 0
        #print(n, np.mean(nom), np.mean(den), np.median(nom), np.median(den))
        #print(n, np.mean(μ), np.std(μ), np.median(μ), np.mean(μ*f), np.std(μ*f), np.median(μ*f))
        #den[den==0] = 0.00001
        up = μ*f
        μ = μ + up
        μ[μ<=0] = 0.00001
        error.append((time.perf_counter()-proctime, np.mean(np.abs(real_image-μ))))
        #print(error[-1])
        if n%100 == 0:
            if real_image is None:
                print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(n, np.sum(up), np.mean(np.abs(real_image-μ)))
            yield μ[:]
    
    yield μ
    yield error


def PWLS(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, use_astra=True): # riviere 2006

    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])

    y = b * np.exp(-proj)


    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
    

    for n in range(iters):
        ŷ = Σj_a_ij(I)
        σ2 = Σj_a_ij(I+ŝ)
        w = y+ŝ+σ2/(G*G*E*E)
        ñ = 2*Σi_a_ij(w*(y-ŷ))
        nom = ñ - I
        denom = 2*Σi_a_ij(w*(ŷ/I)) + β*v
        I = I+up

        if n%1000 == 0:
            if real_image is None:
                print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(n, np.sum(up), np.mean(np.abs(real_image-I)))

    return I