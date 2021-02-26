import imp
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
import PIL
from scipy.optimize.linesearch import line_search_wolfe1

δ = 0.001
ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
δδψ = lambda x: 1 if x <= δ else 0

def sq_norm(x, _p, _δ):
    return np.sqrt(x**2)
def δsq_norm(x, _p, _δ):
    return x / np.sqrt(x**2)
def δδsq_norm(x, _p, _δ):
    return np.ones_like(x)

def p_norm(x, p, δ):
    a = (2*δ)**(-p)*(δ*δ*p)**(p-1)
    b = δ*(1-0.5*p)
    f = np.bitwise_and(x<δ, x>-δ)
    r = np.zeros_like(x)
    r[f] = a*x[f]**2
    r[~f] = (np.abs(x[~f]-b*np.sign(x[~f]))**p) / p
    return r

def δp_norm(x, p, δ):
    a = (2*δ)**(-p)*(δ*δ*p)**(p-1)
    b = δ*(1-0.5*p)
    f_pos = x>=0
    xp = x[f_pos]
    xn = x[~f_pos]

    f_pos_gδ = np.bitwise_and(f_pos, x>δ)
    f_pos_ngδ = np.bitwise_and(f_pos, x<=δ)
    f_pos_gδ_gb = np.bitwise_and(f_pos_gδ, x>=-b)
    f_pos_gδ_ngb = np.bitwise_and(f_pos_gδ, x<-b)

    f_npos_gδ = np.bitwise_and(~f_pos, x<-δ)
    f_npos_ngδ = np.bitwise_and(~f_pos, x>=-δ)
    f_npos_gδ_gb = np.bitwise_and(f_npos_gδ, x>=-b)
    f_npos_gδ_ngb = np.bitwise_and(f_npos_gδ, x<-b)
    
    r = np.zeros_like(x)
    r[f_pos_ngδ] = 2*a*x[f_pos_ngδ]
    r[f_npos_ngδ] = 2*a*x[f_npos_ngδ]

    if p==1:
        r[f_pos_gδ_gb] = 1
        r[f_pos_gδ_ngb] = 1
        r[f_npos_gδ_gb] = 1
        r[f_npos_gδ_ngb] = 1
    else:
        r[f_pos_gδ_gb] = (x[f_pos_gδ_gb]-b)**(p-1)
        r[f_pos_gδ_ngb] = (-x[f_pos_gδ_ngb]+b)**(p-1)
        r[f_npos_gδ_gb] = (x[f_npos_gδ_gb]+b)**(p-1)
        r[f_npos_gδ_ngb] = (-x[f_npos_gδ_ngb]-b)**(p-1)

    r[r<0] = 0
    return r

def δδp_norm(x, p, δ):
    a = (2*δ)**(-p)*(δ*δ*p)**(p-1)
    b = δ*(1-0.5*p)
    f_pos = x>=0

    f_pos_gδ = np.bitwise_and(f_pos, x>δ)
    f_pos_ngδ = np.bitwise_and(f_pos, x<=δ)
    f_pos_gδ_gb = np.bitwise_and(f_pos_gδ, x>=-b)
    f_pos_gδ_ngb = np.bitwise_and(f_pos_gδ, x<-b)

    f_npos_gδ = np.bitwise_and(~f_pos, x<-δ)
    f_npos_ngδ = np.bitwise_and(~f_pos, x>=-δ)
    f_npos_gδ_gb = np.bitwise_and(f_npos_gδ, x>=-b)
    f_npos_gδ_ngb = np.bitwise_and(f_npos_gδ, x<-b)
    
    r = np.zeros_like(x)
    r[f_pos_ngδ] = 2*a
    r[f_npos_ngδ] = 2*a

    if p!=1:
        if p==2:
            r[f_pos_gδ_gb] = (p-1)
            r[f_pos_gδ_ngb] = (p-1)
            r[f_npos_gδ_gb] = (p-1)
            r[f_npos_gδ_ngb] = (p-1)
        else:
            r[f_pos_gδ_gb] = (p-1)*(x[f_pos_gδ_gb]-b)**(p-2)
            r[f_pos_gδ_ngb] = (p-1)*(-x[f_pos_gδ_ngb]+b)**(p-2)
            r[f_npos_gδ_gb] = (p-1)*(x[f_npos_gδ_gb]+b)**(p-2)
            r[f_npos_gδ_ngb] = (p-1)*(-x[f_npos_gδ_ngb]-b)**(p-2)

    r[r<0] = 0
    return r

from PotentialFilter import potential_filter as c_ψ
from PotentialFilter import potential_dx_filter as c_δψ
from PotentialFilter import potential_dxdx_filter as c_δδψ
from PotentialFilter import potential_dx_t_filter as c_δψ_t
from PotentialFilter import square_filter as c_sq
from PotentialFilter import square_dx_filter as c_δsq
from PotentialFilter import square_dxdx_filter as c_δδsq
from PotentialFilter import mod_p_norm_filter as c_p_norm
from PotentialFilter import mod_p_norm_dx_filter as c_δp_norm
from PotentialFilter import mod_p_norm_dx_t_filter as c_δp_t_norm
from PotentialFilter import mod_p_norm_dxdx_filter as c_δδp_norm
from PotentialFilter import edge_preserving_filter as c_ψ_edge
from PotentialFilter import edge_preserving_dx_filter as c_δψ_edge
from PotentialFilter import edge_preserving_dx_t_filter as c_δψ_t_edge


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
            if n%10 == 0:
                if real_image is None:
                    print(n, np.mean(Δμ), np.std(Δμ), np.median(Δμ), np.sum(Δμ))
                else:
                    print(n, np.sum(Δμ), np.sum(np.abs(real_image-μ)))


        #print("iteration finished: ", time.process_time()-proctime, "s change:", np.mean(Δμ))
    
    return μ


def print_stats(i, x):
    print(i, np.mean(x), np.std(x), np.median(x), np.sum(x))

def logLikelihood(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**3, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True):
    y = b*np.exp(-proj)
    
    r= 0
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
        Σj_a_ij2 = utils.Ax2_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        Σi_a_ij2 = lambda x: tigre.Atb(x, geo, angles)
        Σj_a_ij2 = lambda x: tigre.Ax(x, geo, angles)


    error = []
    obj_func = []
    proctime = time.perf_counter()
    #γ = Σj_a_ij(np.ones_like(μ))
    #if(np.count_nonzero(γ==0)>0): raise Exception("Some Detector pixels are always empty")

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
    f = -L+β*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]

    for i in range(iters):
        l = Σj_a_ij(μ)
        ŷ = b*np.exp(-l)
        L = Σi_a_ij(y*(np.log(b)-l)-ŷ)
        R = 0.5*c_μm(μ, c_sq)
        Ṙ = c_μm(μ, δψ, p=p)
        R̈ = c_μm(μ, δδψ, p=p)
        L̇ = Σi_a_ij( y-ŷ )
        L̈ = -Σi_a_ij2(ŷ)
        δf = -L̇ + β*Ṙ
        δδf = -L̈ + β*R̈
        #δf[δδf==0] = 0
        δδf[δδf==0] = 1
        μ = μ + 0.6 * (δf / δδf)
        μ[μ<0] = 0
        #μ[~np.bitwise_and(μ>0, μ<np.inf)] = 0
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        f = -L+β*R
        obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
        if i%10 == 0:
            yield μ[:]
    yield μ
    yield (error, obj_func, μ)

def PIPLE_old(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, βp=10**3, βr=10**3, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True): # stayman 2013

    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.array(initial[:])
    #λ = 0 # registration parameters (3 angles, 3 translations)
    #H = σI
    r = 0.0

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
    ε = 0
    c[c<ε] = ε
    d = Σi_a_ij2(c)
    #print(np.mean(d), np.median(d), np.min(d), np.max(d))
    error = []
    obj_func = []
    proctime = time.perf_counter()

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
    f = -L+βr*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]
    for n in range(iters):
        #for r in range(1, R):
        #    H[r] = BFGS()
        #    Δθ = (ψp*δW(λ)*μp).T * δf(ψp*(μ-W(λ)*μp))
        #    eφ = λ[r-1] + φ * H[r] * ΔΘ(λ[r-1])
        #    λ[r] = λ[r-1] + eφ * H[r] * ΔΘ(λ[r-1])
        #λ[0] = λ[-1]
        #H[0] = H[-1]

        l = Σj_a_ij(μ)
        ḣ = b*np.exp(-l)-y
        #nz_l = l>0
        #h0 = (y[nz_l]*np.log(b)-b)
        #p1 = b*np.exp(-l[nz_l])
        #p1[p1==0] = 0.00000001
        #hl = (y[nz_l]*np.log(p1)-b*np.exp(-l[nz_l]))
        #c = np.zeros_like(l)
        #c[nz_l] = 2*(h0 - hl + ḣ[nz_l]*l[nz_l]) / (l[nz_l]**2)
        #ḧ = b
        #c[~nz_l] = ḧ
        #c[c<0] = 0
        #d = Σi_a_ij2(c)
        
        #ḣ = (y / ( b*np.exp(-l̂) + r ) - 1)*b*np.exp(-l̂)
        #print(np.mean(ḣ), np.median(ḣ), np.min(ḣ), np.max(ḣ))
        #p_norm = c_μm(μ, c_δp_norm, p=p)
        #print(np.mean(p_norm), np.median(p_norm), np.min(p_norm), np.max(p_norm))
        #dp_norm = c_μm(μ, c_δδp_norm, p=p)
        #print(np.mean(dp_norm), np.median(dp_norm), np.min(dp_norm), np.max(dp_norm))

        denom = d + βr * c_μm(μ, δδψ, p=p)
        up = (
            Σi_a_ij(ḣ) \
            #- βp * δfr * (Ψr*μ)  \
            - βr * c_μm(μ, δψ, p=p)
        ) / (
            denom
        )
        up[denom==0] = 0
        μ = μ + up
        μ[μ<0] = 0
        
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        L = Σi_a_ij(y*(np.log(b)-l)-ŷ)
        R = 0.5*c_μm(μ, c_sq)
        f = -L+βr*R
        obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
        if n%10 == 0:
            #if real_image is None:
            #    print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            #else:
            #    print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
        elif n == iters-2:
            yield μ
        #print(n, np.mean(μ), np.std(μ), np.median(μ), np.min(μ), np.max(μ))
    yield μ
    yield (error, obj_func, μ)

def CCA(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**2, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, M=-1, use_astra=True): # fessler 1995
    y = np.ma.masked_array(b*np.exp(-proj), mask=np.ma.nomask)
    
    if initial is None:
        if use_astra:
            initial = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
        else:
            initial = tigre.algorithms.fdk(proj, geo, angles) # initial guess
        
    μ = np.array(initial[:])

    r = 0.1
    ω = 0.6

    if M==-1:
        M = len(angles)
    
    if M!=0:
        m_size = math.ceil(len(proj)/M)
        print(M, m_size, len(proj)/M)

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        #Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    error = []
    obj_func = []
    proctime = time.perf_counter()
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    f = -L+β*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]
    all_vectors = geo['Vectors']
    for i in range(iters):
        for m in range(0, M):
            if(M > 1):
                geo['Vectors'] = all_vectors[m:m+1]
                Σj_a_ij = utils.Ax_astra(out_shape, geo)
                Σi_a_ij = utils.Atb_astra(out_shape, geo)
                Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
                s = slice(m*m_size, (m+1)*m_size, 1)
                y_s = y[:,s]
            else:
                y_s = y

            l = Σj_a_ij( μ )

            ȳ = b*np.exp(-l)
            L = Σi_a_ij(y_s*(np.log(b)-l)-ȳ)
            R = 0.5*c_μm(μ, c_sq)
            #L̇ = Σi_a_ij( (1-y/ȳ) * b * np.exp(-l) )
            #part = ȳ - r - y + r*y/ȳ
            L̇ = Σi_a_ij( (1-y_s/(ȳ+r))*ȳ)

            L̈ = Σi_a_ij2( (1-y_s*r/( (ȳ+r)*(ȳ+r) )) * ȳ )
            #L̈ = Σi_a_ij2( part + y*r*r/(ȳ*ȳ) )

            Ṗ = c_μm(μ, δψ, p=p)
            P̈ = c_μm(μ, δδψ, p=p)

            nom = (L̇ - β * Ṗ)
            den = (L̈ + β * P̈)

            #print_stats(i, l)
            #print_stats(i, ȳ)
            #print_stats(i, L̇)
            #print_stats(i, L̈)
            #print_stats(i, Ṗ)
            #print_stats(i, P̈)
            up = ω*nom/den
            μ = μ + up
            μ[μ<0] = 0
            
            if(M > 1):
                Σi_a_ij.free()
                Σi_a_ij2.free()
                Σj_a_ij.free()

        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        f = -L+β*R
        obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))


        if i%10 == 0:
            #if real_image is None:
            #    print(i, np.mean(up), np.std(up), np.median(up), np.sum(up))
            #else:
            #    print(i, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
    yield μ
    yield (error, obj_func, μ)

def PL_PSCD(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**4, β=10**2, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True): # erdogan 1998

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
    μ = np.array(initial[:])

    c = (1-y*r/((b+r)*(b+r)))*b
    ε = 0.00001
    c[c<ε] = ε
    d = Σi_a_ij2(c)

    error = []
    obj_func = []
    proctime = time.perf_counter()

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
    f = -L+β*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]
    for n in range(iters):

        l̂ = Σj_a_ij(μ)
        ḣ = (y/(b*np.exp(-l̂)+r)-1)*b*np.exp(-l̂)
        #q̇ = ḣ(l̂) + c*(l̂-l)
        #q̇ = (y/(b*np.exp(-l̂)+r)-1)*b*np.exp(-l̂)
        q̇ = ḣ
        Ṙ = c_μm(μ, δψ, p=p)
        p̂ = c_μm(μ, δδψ, p=p)

        Q̇ = Σi_a_ij(q̇)
        μ_old = μ[:]
        for _ in range(5):
            μ = μ - (Q̇ + d*(μ-μ_old) + β*Ṙ) / (d+β*p̂)
            μ[μ<0] = 0
        #μ[μ<0] = 0
        #inp = np.zeros_like(μ)
        #inp[j] = 1
        #row = Σj_a_ij(inp)
        #q̇ = q̇ + row*c
        #den = Σj_a_ij(μ-μ_old)*c
        #den_0 = den != 0
        #q̇[den_0] = q̇[den_0] / den[den_0]
        #l̂ = l̂+(q̇-ḣ)/c

        if n%10==0:
            β = β*0.1

        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        L = Σi_a_ij(y*(np.log(b)-l)-ŷ)
        R = 0.5*c_μm(μ, c_sq)
        f = -L+β*R
        obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
        if n%10 == 0:
            #up = μ-μ_old
            #if real_image is None:
            #    print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            #else:
            #    print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
        #if n == iters-2:
        #    yield μ

    yield μ
    yield (error, obj_func, μ)

def ML_OSTR(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**2, β=10**2, use_astra=True): # erdogan 1999
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
    obj_func = []
    proctime = time.perf_counter()

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
    f = -L+β*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]
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
        L = Σi_a_ij(y*(np.log(b)-l)-ŷ)
        R = 0.5*c_μm(μ, c_sq)
        f = -L+β*R
        obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
        if it%10 == 0:
            #if real_image is None:
            #    print(it, np.mean(up), np.std(up), np.median(up), np.sum(up))
            #else:
            #    print(it, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
    yield μ
    yield (error, obj_func, μ)

def PL_OSTR(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**2, β=10**2, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True): # erdogan 1999

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

    γ = Σj_a_ij(np.ones_like(μ))
    d = Σi_a_ij( γ * (y-r)**2 / y)
    error = []
    obj_func = []
    proctime = time.perf_counter()

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
    f = -L+β*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]
    for n in range(iters):
        for _m in range(M):
            l̂ = Σj_a_ij(μ)
            ŷ = b*np.exp(-l̂)
            ḣ = (y / ( ŷ + r )-1)*ŷ

            L̇ = M * Σi_a_ij(ḣ)
            nom = (L̇ + β * c_μm(μ, δψ, p=p) )
            den = (d + 2*β* c_μm(μ, δδψ, p=p) )
            den[den==0] = 0.00001
            #if iter%10 == 0:
            #print(n, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
            #print(n, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )
            up = nom/den
            μ = μ - up
            μ[μ<0] = 0
            error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
            L = Σi_a_ij(y*(np.log(b)-l)-ŷ)
            R = 0.5*c_μm(μ, c_sq)
            f = -L+β*R
            obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
            if n%10 == 0:
                #if real_image is None:
                #    print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
                #else:
                #    print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
                yield μ[:]

    yield μ
    yield (error, obj_func, μ)

def PL_C(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**2, β=10**3, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True): # aootaphao 2008

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

    error = []
    obj_func = []
    proctime = time.perf_counter()

    l = Σj_a_ij(μ)
    ŷ = b*np.exp(-l)
    L = Σi_a_ij((y*(np.log(b)-l)-ŷ))
    R = 0.5*c_μm(μ, c_sq)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
    f = -L+β*R
    obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
    yield μ[:]
    for n in range(iters):
        #l = Σj_a_ij(μ)
        #ups = pool.imap_unordered(update_step, [(y[i::subsets], μ, β, b, out_shape, geos[i]) for i in range(subsets)] )
        #up = functools.reduce(lambda x,y: x+y, ups) / subsets
        l = Σj_a_ij(μ)
        Ṗ = β*c_μm(μ, δψ, p=p)
        P̈ = μ*β*c_μm(μ, δδψ, p=p)
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
        μ = μ - up
        μ[μ<=0] = 0.00001
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        L = Σi_a_ij(y*(np.log(b)-l)-ŷ)
        R = 0.5*c_μm(μ, c_sq)
        f = -L+β*R
        obj_func.append((time.perf_counter()-proctime, np.sum(f[~np.bitwise_and(f>0, f<np.inf)])))
        if n%10 == 0:
            #if real_image is None:
            #    print(n, np.mean(up), np.std(up), np.median(up), np.sum(up))
            #else:
            #    print(n, np.sum(up), np.sum(np.abs(real_image-μ)))
            yield μ[:]
    
    yield μ
    yield (error, obj_func, μ)

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
        curv = np.ones_like(deriv)*wi
    else:
        curv = wi
    return deriv, curv

def calc_oc(yi, bi, ri, li):
    h = lambda y,b,r,l: y * np.log(b*np.exp(-l)+r) - (b*np.exp(-l)+r)
    h0 = lambda y,b,r: y * np.log(b+r) - b+r
    dh = lambda y,b,r,l: (1 - y / (b*np.exp(-l)+r)) * b*np.exp(-l)
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
        il0 = li < 0.1

    i = ~il0
    tmp = h(yi[i],bi,ri,li[i]) - h0(yi[i],bi,ri) - li[i] * dh(yi[i],bi,ri,li[i])
    tmp[tmp<0] = 0
    ni[i] = 2 / li[i]**2 * tmp

    return ni


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
    
    #% Compute optimal surrogate parabola curvatures
    #% for Poisson transmission model based on Erdogan's formula.
    if curvtype=='oc':
        ni = calc_oc(yi, bi, ri, li)
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

def pl_iot(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**2, β=10**3, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True, M=1):

    #%function [xs, info] = pl_iot(x, Ab, data, R, [options])
    #%|
    #%| Generic penalized-likelihood minimization,
    #%| for arbitrary negative log-likelihood with convex non-quadratic penalty,
    #%| via incremental optimization transfer using separable quadratic surrogates.
    #%|
    #%| cost(x) = sum_i h(data_i; [Ax]_i) + R(x),
    #%|
    #%| in
    #%|	x	[np 1]		initial estimate
    #%|	Ab	[nd np]		system matrix (see Gblock)
    #%|	data	{cell}		whatever data is needed for the likelihood
    #%|				data{1} must have size "[nb,na]", so for 3D
    #%|				one must use reshaper(yi, '2d').
    #%|	R			penalty object (see Robject.m)
    #%|
    #%| options
    #%|	dercurv	{function_handle} function returning derivatives and curvatures
    #%|				of negative log-likeihood via the call:
    #%|		[deriv curv] = dercurv(data, Ab{m}*x, curvtype, iblock, nblock)
    #%|				or choose 'trl' or 'wls' (default)
    #%|	niter 	#		# total iterations (default: 1+0, pure OS!)
    #%|	os	#		how many "warmup" OS iterations (default: 1)
    #%|	riter	#		# of penalty subiterations (default: 3).
    #%|	curvtype ''		curvature type:
    #%|					'pc' : precomputed (default)
    #%|					'oc' : optimal (ensures monotonic)
    #%|	pixmax	(value)		upper bound on pixel values (default: infinity)
    #%|	pixmin	(value)		lower bound on pixel values (default: 0)
    #%|	isave	[]		list of iterations to archive
    #%|				(default: [] 'last)
    #%|	gi	[nd]		precomputed Ab * 1 factors.
    #%|	chat	(value)		verbosity (default: 0)
    #%|	userfun			user defined function handle (see default below)
    #%|
    #%| out
    #%|	xs	[np nsave]	estimates each (saved) iteration
    #%|	info	[niter+1 ?]	userfun output.  default is cpu times
    #%|
    #%| Copyright 2005-3-8, Jeff Fessler, University of Michigan


    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)


    Ab = Σj_a_ij
    Atb = Σi_a_ij
    At2b = Σi_a_ij2
    x = np.array(initial)
    nblock = 1
    error = []
    obj_func = []
    proctime = time.perf_counter()

    #% defaults
    dercurv = 'wls'
    curvtype = 'pc'
    riter = 3
    os = 2
    niter = iters #% default is pure OS, no IOT
    pixmin = 0
    pixmax = np.inf
    gi = []
    update_even_if_denom_0 = True
    chat = False

    data = (proj, b*np.ones_like(proj), np.zeros_like(proj))

    Rdenom = lambda μ: c_μm(μ, δδψ, p=p)
    Rcgrad = lambda μ: c_μm(μ, δψ, p=p)

    #%
    #% inner_update()
    #% given SPS for likelihood, update x using regularization subiterations
    #%
    def inner_update(x, lnum, ldenom):
        for _ in range(riter):
            rdenom = Rdenom(x);
            num = lnum - Rcgrad(x) + rdenom * x;
            
            #% update
            if update_even_if_denom_0:
                x = num / (ldenom + rdenom);
            else:
                old = x[ldenom == 0]
                x = num / (ldenom + rdenom);
                x[ldenom == 0] = old;

            x[~(x>pixmin)] = pixmin	#% lower bound
            x[~(x<pixmax)] = pixmax	#% upper bound
        return x

    #% options
    #arg = vararg_pair(arg, varargin);
    #arg.isave = iter_saver(arg.isave, arg.niter);

    if dercurv=='wls':
        dercurv = dercurve_wls

    elif dercurv== 'trl':
        dercurv = dercurve_trl

    #[nb, na] = proj.shape

    #% g_i = sum_j g_ij.  caution: this requires real g_ij and g_ij >= 0
    if len(gi)==0:
        gi = Ab(np.ones_like(x))
        #gi = max(reale(arg.gi), 0) #% trick: just in case...
        #gi = reshape(arg.gi, [nb na])

    npixel = x.shape;
    #% precompute likelihood-term denominator if needed
    if curvtype == 'pc':
        [_, curvi] = dercurv(data, 0, 'pc')
        ldenom = Atb(gi * curvi) #% one denominator shared by all subsets
        ldenoms = np.zeros((*npixel, nblock));
    else:
        #ldenoms = np.zeros_like(np, nblock);
        ldenoms = np.zeros((*npixel, nblock));
        ldenom = np.zeros(npixel)

    x[~(x<pixmax)] = pixmax
    x[~(x>pixmin)] = pixmin
    
    #%
    #% precompute gradient-related state vectors, usually by OS-SPS warmup iterations
    #%
    vvm = np.zeros((*npixel,nblock));

    #%
    #% SPS-OS iterations to "warm up" (often 1 or 2 iterations suffices)
    #%
    for it in range(os):
        for iset in range(nblock):
            #ticker([mfilename, ': os'], [it, iset], [os, nblock])

            #iblock = starts(iset);
            iblock = 0
            #ia = iblock:nblock:na;

            #li = Ab{iblock} * x;	% l=A*x "line integrals"
            li = Ab(x)
            #li = reshape(li, nb, length(ia));
            #[dhi curvi] = feval(arg.dercurv, data, li, arg.curvtype, iblock, nblock);
            [dhi, curvi] = dercurv(data, li, curvtype, iblock, nblock)

            if curvtype != 'pc':# % on-the-fly curvatures
                ldenoms[...,iblock] = Atb(gi * curvi);
                ldenom = nblock * ldenoms[...,iblock];
                ldenom = Atb(gi * curvi)                

            lnum = ldenom * x - nblock * Atb(dhi)

            if it == os: #% save last gradient-related state vectors
                vvm = lnum / nblock

            x = inner_update(x, lnum, ldenom);
        if it%10==0:
            yield x
        
        #info.append(userfun())
        if chat: print('Range {} {}', min(x), max(x))

        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-x))))
        obj_func.append((time.perf_counter()-proctime, 0))
    
    #%
    #% At this point we have initialized vv_m and ldenoms
    #% and initialized x based on the last subset.
    #% It is often logical to update x using all subsets now.
    #% This is "almost free" since no new likelihood gradients are used.
    #%
    vv = np.sum(vvm,-1);
    if curvtype == 'pc':
        ldenoms = ldenom / nblock;
    else:
        ldenom = np.sum(ldenoms,1);
    x = inner_update(x, vv, ldenom)
    error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-x))))
    obj_func.append((time.perf_counter()-proctime, 1))
    yield x
    #%
    #% IOT iterations
    #%
    for it in range(os, niter):
        for iset in range(nblock):
            #ticker([mfilename ': iot'], [iter iset], [arg.niter nblock])

            #iblock = starts(iset);
            iblock = 0
            #ia = iblock:nblock:na;

            #li = Ab{iblock} * x;	% l=A*x "line integrals"
            li = Ab(x)	#% l=A*x "line integrals"
            #li = reshape(li, nb, length(ia));
            [dhi, curvi] = dercurv(data, li, curvtype, iblock, nblock)

            if curvtype != 'pc':
                ldenom = ldenom - ldenoms[...,iblock]
                ldenom[...,iblock] = Atb(gi * curvi)
                ldenom = ldenom + ldenoms[...,iblock]

            vv = vv - vvm[...,iblock];
            if curvtype=='pc':
                vvm[...,iblock] = ldenoms * x - Atb(dhi)
            else:
                vvm[...,iblock] = ldenoms[...,iblock] * x - Atb(dhi);

            vv = vv + vvm[...,iblock]

            x = inner_update(x, vv, ldenom);
        
        if it%10==0:
            yield x
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-x))))
        obj_func.append((time.perf_counter()-proctime, 2))

    yield error, obj_func, x
    
def pl_pcg_qs_ls(proj, out_shape, geo, angles, iters, initial=None, real_image=None, b=10**2, β=10**3, p='quad_wls', δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True, M=1):

    #%|function [xs, info] = pl_pcg_qs_ls(x, A, data, dercurv, R, varargin)
    #%|
    #%| Unconstrained generic penalized-likelihood minimization,
    #%| for arbitrary negative log-likelihood with convex non-quadratic penalty,
    #%| via preconditioned conjugate gradient algorithm
    #%| with quadratic surrogate based line search.
    #%| cost(x) = -log p(data|x) + R(x)
    #%|
    #%| in
    #%|	x	[np 1]		initial estimate
    #%|	A	[nd np]		system matrix
    #%|	data	{cell}		whatever data is needed for the likelihood
    #%|	dercurv	function_handle function returning derivatives and curvatures
    #%|				of negative log-likeihood via:
    #%|				[deriv curv] = dercurv(data, A*x, curvtype)
    #%|	R			penalty object (see Robject.m)
    #%|
    #%| options (name / value pairs)
    #%|	niter	?		# total iterations
    #%|	isave	[]		list of iterations to archive
    #%|					(default: [] 'last')
    #%|	stepper	?		method for step-size line search
    #%|					default: {'qs', 3}
    #%|	precon	[np np]		preconditioner, matrix | object; default: 1
    #%|	userfun			user defined function handle (see default below)
    #%|	curvtype		type of curvature, default 'pc'
    #%|	restart			restart every so often (default: inf)
    #%|
    #%| out
    #%|	xs	[np,nsave]	estimates each (saved) iteration
    #%|	info	[niter+1 ?]	userfun output. default is: gamma, step, time
    #%|
    #%| Copyright 2004-2-1, Jeff Fessler, University of Michigan

    Σj_a_ij = utils.Ax_astra(out_shape, geo)
    Σi_a_ij = utils.Atb_astra(out_shape, geo)
    Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    

    Ab = Σj_a_ij
    Atb = Σi_a_ij
    At2b = Σi_a_ij2
    x = np.array(initial)
    nblock = 1
    error = []
    obj_func = []
    proctime = time.perf_counter()

    stepper = {}
    step0 = 0 #% for backward compatability, but maybe 1 makes sense sometimes
    niter = iters
    precon = 1
    restart = np.inf #% restart every this many iterations (default: never)
    curvtype = 'pc'

    #arg = vararg_pair(arg, varargin);

    if len(stepper) == 0:
        stepper = ['qs', 3] #% quad surr with this # of subiterations
    #%info = zeros(arg.niter,?); #% trick: do not initialize since size may change
    #Rdercurv = lambda μ: [c_μm(μ, δψ, p=p), c_μm(μ, δδψ, p=p)]
    pot_quad = lambda C1x: np.ones_like(C1x)
    def pot_huber(C1x, d = 0.001):
        w = np.ones_like(C1x)
        ii = np.abs(C1x) > d
        w[ii] = d / np.abs(C1x[ii])
        return w

    pot, dercurv = p.split('_')
    if pot=="huber": 
        pot = pot_huber
    else:
        pot = pot_quad

    def Rdercurv(C1x):
        deriv = β * pot(C1x) * C1x
        curv = β * pot(C1x)
        return deriv, curv

    if dercurv=='wls':
        dercurv = dercurve_wls
    elif dercurv == 'tlr':
        dercurv = dercurve_trl
        curvtype = 'oc'

    if np.isscalar(b):
        data = (proj, b*np.ones_like(proj), np.zeros_like(proj))
    else:
        data = (proj, b, np.zeros_like(proj))
    #% initialize projections
    Ax = Ab(x)
    def C1(x):
        C1x = np.zeros_like(x)
        C1x[:-1] = x[:-1]-x[1:]
        C1x[:,:-1] += x[:,:-1]-x[:,1:]
        C1x[:,:,:-1] += x[:,:,:-1]-x[:,:,1:]
        C1x /= 3
        return C1x
    C1x = C1(x)

    oldinprod = 0

    #% iterate
    warneddir = False
    warnedstep = 0
    for it in range(niter):

        #% gradient of cost function
        [hderiv, hcurv] = dercurv(data, Ax, curvtype)
        [pderiv, pcurv] = Rdercurv(x)
        grad = Atb(hderiv) + C1(pderiv);

        #% preconditioned gradient
        pregrad = precon * grad;

        #% direction
        newinprod = grad.flatten().dot(pregrad.flatten())
        if oldinprod == 0 or it%restart == 0:
            ddir = -pregrad;
            gamma = 0;
        else:
            #% todo: offer other step-size rules ala hager:06:aso
            gamma = newinprod / oldinprod; #% Fletcher-Reeves
            ddir = -pregrad + gamma * ddir;

        oldinprod = newinprod;

        #% check if descent direction
        if ddir.flatten().dot(grad.flatten()) > 0:
            if not warneddir: #% todo: warn every time!
                warneddir = True;
                print('wrong direction so resetting')
                print('<ddir,grad>=%g, |ddir|=%g, |grad|=%g', ddir.flatten().dot(grad.flatten()), np.linalg.norm(ddir.flatten()), np.linalg.norm(grad.flatten()))

            #% reset
            ddir = -pregrad;
            oldinprod = 0;

        #% step size in search direction
        Adir = Ab(ddir);
        C1dir = C1(ddir); #% caution: can be a big array for 3D problems

        #% multiple steps based on quadratic surrogates
        if stepper[0] == 'qs':
            nsub = stepper[1]
            step = step0
            for i_s in range(nsub):
                if step != 0:
                    [hderiv, hcurv] = dercurv(data, Ax + step * Adir, curvtype)
                    [pderiv, pcurv] = Rdercurv(C1x + step * C1dir)
                denom = (Adir**2).flatten().dot(hcurv.flatten()) + (C1dir**2).flatten().dot(pcurv.flatten())
                numer = Adir.flatten().dot(hderiv.flatten()) + C1dir.flatten().dot(pderiv.flatten())
                if denom == 0:
                    print('found exact solution???  step=0 now!?')
                    step = 0;
                else:
                    step = step - numer / denom

                if step < 0:
                    if not warnedstep:
                        print('downhill step?')
                        print('iter=%d step=%g', it, step)

        #% update
        Ax = Ax + step * Adir;
        C1x = C1x + step * C1dir;
        x = x + step * ddir;

        if it%10==0:
            yield x
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-x))))
        obj_func.append((time.perf_counter()-proctime, 2))

    yield error, obj_func, x

def PIRPLE(proj, out_shape, geo, angles, iters, initial, real_image, b=10**4, βp=10**3, βr=10**3, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True): # stayman 2013

    μ = np.array(initial)
    λ = np.zeros((3,4), dtype=float)
    λ[0,0] = 1
    λ[1,1] = 1
    λ[2,2] = 1
    H = np.eye(len(λ))
    I = np.eye(len(λ))
    R = 5

    Σj_a_ij = utils.Ax_astra(out_shape, geo)
    Σi_a_ij = utils.Atb_astra(out_shape, geo)
    Σj_a_ij2 = utils.Ax2_astra(out_shape, geo)
    Σi_a_ij2 = utils.At2b_astra(out_shape, geo)

    r = 0.0

    proctime = time.perf_counter()
    error = []
    obj_func = []

    y = b*np.exp(-proj) + r
    
    c = (1-y*r/((b+r)*(b+r)))*b
    ε = 0
    c[c<ε] = ε
    d = Σi_a_ij2(c)

    def C1(x):
        C1x = np.zeros_like(x)
        C1x[:-1] = x[:-1]-x[1:]
        C1x[:,:-1] += x[:,:-1]-x[:,1:]
        C1x[:,:,:-1] += x[:,:,:-1]-x[:,:,1:]
        C1x /= 3
        return C1x

    def W(μ, λ):
        mat = np.zeros((4,4))
        mat[:3,:] = λ.reshape((3,4))
        mat[3,3] = 1
        return scipy.ndimage.affine_transform(μ, mat)

    def θ(μ, λ):
        return c_μm(μ-W(real_image, λ), δψ, p=p)

    def gradΘ(μ, λ):
        λ = λ.reshape((3,4))
        res = np.zeros_like(λ)
        Wμp = W(real_image, λ)
        s = c_μm(Wμp,δψ, p=p)
        for i in range(len(λ)):
            λi = np.array(λ[:])
            λi[i] += 0.1
            res[i] = ((c_μm(W(real_image,λi), δψ, p=p)-s)/0.1).flatten().dot(C1(c_μm(μ-Wμp, δψ, p=p)).flatten())
        return res

    old_fval = θ(μ, λ)
    gfk = gradΘ(μ, λ)
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    pk = -np.dot(H, gfk)
    alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                line_search_wolfe1(lambda λ: θ(μ, λ).flatten(), lambda λ: gradΘ(μ, λ).flatten(), λ.flatten(), pk.flatten(), gfk.flatten(), old_fval.flatten(), old_old_fval.flatten(), amin=1e-100, amax=1e100)


    for it in range(iters):
        for r in range(R):
            # ∇λΘ(λ, μ)
            gfk = gradΘ(μ, λ)
            # BFGS update of H
            
            xkp1 = λ + alpha_k * pk
            
            sk = xkp1 - λ
            λ = xkp1
            if gfkp1 is None:
                gfkp1 = gradΘ(μ, xkp1)

            yk = gfkp1 - gfk
            gfk = gfkp1

            rhok_inv = np.dot(yk, sk)
            # this was handled in numeric, let it remaines for more safety
            if rhok_inv == 0.:
                rhok = 1000.0
            else:
                rhok = 1. / rhok_inv
            A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
            A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
            H = np.dot(A1, np.dot(H, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])
            # line search ϕ

            pk = -np.dot(H, gfk)
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search_wolfe1(lambda λ: θ(μ, λ), lambda λ: gradΘ(μ, λ), λ, pk, gfk, old_fval, old_old_fval, amin=1e-100, amax=1e100)
            # λ = λ + ϕ*H*∇λΘ(λ)
            λ = λ + alpha_k * pk

        l = Σj_a_ij(μ)
        ḣ = b*np.exp(-l)-y
        
        Wμp = W(real_image, λ)
        
        up = (
            Σi_a_ij(ḣ) \
            - βr * c_μm(μ, δψ, p=p) \
            - βp * c_μm(μ-Wμp, δψ, p=p)
        ) / (
            d \
            + βr * c_μm(μ, δδψ, p=p) \
            + βp * c_μm(μ-Wμp, δδψ, p=p)
        )
        μ = μ + up
        μ[~(μ>0)] = 0

        if it%20==0:
            yield μ
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        obj_func.append((time.perf_counter()-proctime, 2))

    yield error, obj_func, μ

def PIPLE(proj, out_shape, geo, angles, iters, initial, real_image, b=10**4, βp=10**3, βr=10**3, p=1, δψ=c_δp_norm, δδψ=c_δp_t_norm, use_astra=True): # stayman 2013

    μ = np.array(initial)
    μp = np.array(real_image)
    
    Σj_a_ij = utils.Ax_astra(out_shape, geo)
    Σi_a_ij = utils.Atb_astra(out_shape, geo)
    Σj_a_ij2 = utils.Ax2_astra(out_shape, geo)
    Σi_a_ij2 = utils.At2b_astra(out_shape, geo)

    r = 0.0

    proctime = time.perf_counter()
    error = []
    obj_func = []

    y = b*np.exp(-proj) + r
    
    oc_curv = True
    ε = 0
    if not oc_curv:
        c = (1-y*r/((b+r)*(b+r)))*b
        c[c<ε] = ε
        d = Σi_a_ij2(c)

    def C1(x):
        C1x = np.zeros_like(x)
        C1x[:-1] = x[:-1]-x[1:]
        C1x[:,:-1] += x[:,:-1]-x[:,1:]
        C1x[:,:,:-1] += x[:,:,:-1]-x[:,:,1:]
        C1x /= 3
        return C1x
    C1μp = C1(μp)

    c = np.zeros_like(y)
    for it in range(iters):
        
        l = Σj_a_ij(μ)
        eli = np.exp(-l)
        #ŷ = b*eli + r
        ḣ = y-b*eli
        #ḣ = (1-y/ŷ)*b*eli
        #ḣ = y - ŷ

        if oc_curv:
            f = l>ε
            #c[f] = b*(1-eli[f])
            #c[f] -= y[f]*np.log((b+r)/ŷ[f]) 
            #c[f] += l[f]*b*eli[f]*(y[f]/ŷ[f]-1)
            #c[f] -= y[f]*l[f]
            #c[f] += l[f]*y[f]
            #c[f] *= (2/(l[f]**2))
            #c[f] = 2*b/(l[f]**2) * (1 - eli[f] - l[f]*eli[f])
            #c[~f] = b
            #c[c<ε] = ε
            c = calc_oc(y, b, 0, l)
            #d = Σi_a_ij2(c)
            d = Σi_a_ij(c)

            
        C1μ = C1(μ)
        C1μ_p = C1(μ-μp)
        denom = (d
            #+ βr * c_μm(μ, δδψ, p=p) + βp * c_μm(μ-μp, δδψ, p=p)
            + βr * δδp_norm(C1μ, p, δ) + βp * δδp_norm(C1μ_p, p, δ)
            #+ βr * δδsq_norm(C1μ, p, δ) + βp * δδsq_norm(C1μ_p, p, δ)
            )
        fil = denom==0
        denom[fil]=1
        nom = (
            Σi_a_ij(ḣ)
            #- βr * c_μm(μ, δψ, p=p) - βp * c_μm(μ-μp, δψ, p=p)
            - βr * δp_norm(C1μ, p, δ) - βp * δp_norm(C1μ_p, p, δ)
            #- βr * δsq_norm(C1μ, p, δ) - βp * δsq_norm(C1μ_p, p, δ)
        )
        up = nom / denom
        up[fil] = ε
        
        μ = μ - up
        μ[~(μ>ε)] = ε
        #print(it, np.mean(nom), np.mean(denom), np.mean(up), np.mean(μ))
        #print(it, np.min(nom), np.min(denom), np.min(up), np.min(μ))
        #print(it, np.max(nom), np.max(denom), np.max(up), np.max(μ))
        if it%10==0:
            yield μ
        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-μ))))
        obj_func.append((time.perf_counter()-proctime, 2))

    yield error, obj_func, μ
