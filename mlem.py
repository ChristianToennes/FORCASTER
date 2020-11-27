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
import math

δ = 0.001
ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
δδψ = lambda x: 1 if x <= δ else 0

from PotentialFilter import potential_filter as c_ψ
from PotentialFilter import potential_dx_filter as c_δψ
from PotentialFilter import potential_dxdx_filter as c_δδψ
from PotentialFilter import potential_dx_t_filter as c_δψ_t
from PotentialFilter import square_filter as c_sq
from PotentialFilter import square_dx_filter as c_δsq
from PotentialFilter import square_dxdx_filter as c_δδsq
from PotentialFilter import mod_p_norm_filter as c_p_norm
from PotentialFilter import mod_p_norm_dx_filter as c_δp_norm
from PotentialFilter import mod_p_norm_dxdx_filter as c_δδp_norm


def c_μm(μ, f, δ=0.0005, p=1):
    user_data = ctypes.c_double * 2    
    ptr = ctypes.cast(ctypes.pointer(user_data(δ, p)), ctypes.c_void_p)
    callback = scipy.LowLevelCallable(f(), ptr)
    return scipy.ndimage.generic_filter(μ, callback, size=3)

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
                    print(n, np.sum(Δμ), np.sum(np.abs(real_image-μ)))


        #print("iteration finished: ", time.process_time()-proctime, "s change:", np.mean(Δμ))
    
    return μ


def p_norm(x, p, δ=1):
    δarea = x < δ
    return np.sum(x[δarea])*math.pow(2*δ, -p)*math.pow(δ*δ*2, p-1) + \
           np.sum( np.abs(x[~δarea] - δ*(1-0.5*p) * np.sign(x[~δarea]) ) ** p ) /p

def PIPLE(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, βp=10**4, use_astra=True): # stayman 2013

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
    p = 1.5 # p-norm

    y = b*np.exp(-proj) + r
    N = y.shape

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        #Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    c = (1-y*r/((b+r)*(b+r)))*b
    ε = 0.00001
    c[c<ε] = ε
    d = Σi_a_ij2(c)
    #print(np.mean(d), np.median(d), np.min(d), np.max(d))
    error = []
    proctime = time.perf_counter()
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
        #print(np.mean(ḣ), np.median(ḣ), np.min(ḣ), np.max(ḣ))
        #p_norm = c_μm(μ, c_δp_norm, p=p)
        #print(np.mean(p_norm), np.median(p_norm), np.min(p_norm), np.max(p_norm))
        #dp_norm = c_μm(μ, c_δδp_norm, p=p)
        #print(np.mean(dp_norm), np.median(dp_norm), np.min(dp_norm), np.max(dp_norm))

        up = (
            Σi_a_ij(ḣ) \
            #- βr * Ψr * δfr * (Ψr*μ)  \
            - βp* Ψp * c_μm(μ, c_δp_norm, p)
        ) / (
            d \
            #+ βr * Ψr**2 * ωfr * (ψr*μ) \
            + βp * Ψp**2 * c_μm(μ, c_δδp_norm, p)
        )
        μ = μ + up
        μ[μ<0] = 0
        
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        if n%100 == 0:
            if real_image is None:
                print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
        if n == iters-2:
            yield μ
        #print(n, np.mean(μ), np.std(μ), np.median(μ), np.min(μ), np.max(μ))
    yield μ
    yield error

def CCA(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**2, use_astra=True): # fessler 1995
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
        #Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    error = []
    proctime = time.perf_counter()
    for i in range(iters):
        l = Σj_a_ij( μ )
        ȳ = b*np.exp(-l) + r
        #L̇ = Σi_a_ij( (1-y/ȳ) * b * np.exp(-l) )
        #part = ȳ - r - y + r*y/ȳ
        L̇ = Σi_a_ij( (1-y/ȳ)*b*np.exp(-l))

        L̈ = Σi_a_ij2( (1-y*r/(ȳ*ȳ)) * b * np.exp(-l) )
        #L̈ = Σi_a_ij2( part + y*r*r/(ȳ*ȳ) )

        Ṗ = c_μm(μ, c_δsq)
        P̈ = c_μm(μ, c_δδsq)

        nom = (L̇ - β * Ṗ)
        den = (L̈ + β * P̈)

        #print(i, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
        #print(i, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )
        up = ω*nom/den
        μ = μ + up
        μ[μ<0] = 0
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        
        if i%100 == 0:
            if real_image is None:
                print(i, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(i, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
    yield μ
    yield error

def PL_PSCD(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**2, use_astra=True): # erdogan 1998

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        #Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    r = 0.1
    y = b*np.exp(-proj)
    μ = np.ones(out_shape)*0.1

    c = (1-y*r/((b+r)*(b+r)))*b
    ε = 0.00001
    c[c<ε] = ε
    d = Σi_a_ij2(c)

    error = []
    proctime = time.perf_counter()
    for n in range(iters):

        l̂ = Σj_a_ij(μ)
        ḣ = (y/(b*np.exp(-l̂)+r)-1)*b*np.exp(-l̂)
        #q̇ = ḣ(l̂) + c*(l̂-l)
        #q̇ = (y/(b*np.exp(-l̂)+r)-1)*b*np.exp(-l̂)
        q̇ = ḣ[:]
        Q̇ = Σi_a_ij(q̇)
        Ṙ = c_μm(μ, c_δsq)
        p̂ = c_μm(μ, c_δδsq)

        μ_old = μ[:]
        for _ in range(5):
            μ = μ - (Q̇ + d*(μ-μ_old) + β*Ṙ) / (d+β*p̂)
        μ[μ<0] = 0

        #den = Σj_a_ij(μ-μ_old)*c
        #den_0 = den != 0
        #q̇[den_0] = q̇[den_0] / den[den_0]
        #l̂ = l̂+(q̇-ḣ)/c

        if n%10==0:
            β = β*0.1

        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        if n%100 == 0:
            up = μ-μ_old
            if real_image is None:
                print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
        if n == iters-2:
            yield μ

    yield μ
    yield error

def ML_OSTR(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**2, use_astra=True): # erdogan 1999
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
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)

    #l̂ = Σj_a_ij(μ)
    #print("l", np.mean(l̂), np.median(l̂), np.max(l̂), np.min(l̂))

    d = Σi_a_ij( Σj_a_ij(np.ones_like(μ)) * (y-r)**2 / y)
    d_0 = d==0
    #d[d==0] = 0.000001
    M = 1 # subsets
    error = []
    proctime = time.perf_counter()
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
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        if it%100 == 0:
            if real_image is None:
                print(it, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(it, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
    yield μ
    yield error

def PL_OSTR(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**2, use_astra=True): # erdogan 1999

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
    proctime = time.perf_counter()
    for n in range(iters):
        for _m in range(M):
            l̂ = Σj_a_ij(μ)
            ŷ = b*np.exp(-l̂)
            ḣ = (y / ( ŷ + r )-1)*ŷ
            L̇ = M * Σi_a_ij(ḣ)

            nom = (L̇ + β * c_μm(μ, c_δsq) )
            den = (d + 2*β* c_μm(μ, c_δδsq) )
            den[den==0] = 0.00001
            #if iter%10 == 0:
            #print(n, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
            #print(n, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )
            up = nom/den
            μ = μ - up
            μ[μ<0] = 0
            error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
            if n%100 == 0:
                if real_image is None:
                    print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
                else:
                    print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
                yield μ[:]

    yield μ
    yield error

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
        Ṗ = β*c_μm(μ, c_δsq)
        P̈ = μ*β*c_μm(μ, c_δδsq)
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
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        #print(error[-1])
        if n%100 == 0:
            if real_image is None:
                print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            else:
                print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
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
                print(n, np.sum(up), np.sum(np.abs(real_image-I)))

    return I