from numpy.lib import utils
import tigre
import astra
import numpy as np
import time
import scipy.ndimage
import utils

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


W = np.zeros((3,3,3), dtype=np.bool)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        for k in range(W.shape[2]):
            if i != 1 and j != 1 and k != 1:
                W[i, j, k] = 1.0 / np.sqrt((1-i)*(1-i) + (1-j)*(1-j) + (1-k)*(1-k))

W = W.flatten()

def μm(μ, f):
    def filt(data):
        return np.sum([w*f(data[13]-d) for d,w in zip(data,W)])
    return scipy.ndimage.generic_filter(μ, filt, size=3)

δ = 0.001
ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
δδψ = lambda x: 1 if x <= δ else 0


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

def mlem1(proj, geo, angles, iters, initial=None, p=2):
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

def mlem2(proj, geo, angles, iters, initial=None): # tilley 2017
    if initial is None:
        initial = tigre.algorithms.fdk(proj,geo,angles)
        #initial = np.zeros((geo.nVoxel), dtype=np.float32 )
    y = proj
    μ = initial
    Nμ = initial.shape # Number of voxels
    Ny = proj.shape # Number of measurements
    
    B = np.zeros((Ny, Ny)) # Gain/blur matrix
    Bt = np.zeros((Ny, Ny))
    Bs = np.zeros((Ny, Ny)) # spot blur
    Bst = np.zeros((Ny, Ny))
    Bd = np.zeros((Ny, Ny)) # scintillator blur
    Bdt = np.zeros((Ny, Ny))
    G = np.zeros((Ny, Ny)) # data scale by bare-beam photon flux
    Gt = np.zeros((Ny, Ny))
    K = np.zeros((Ny, Ny)) # Measurement covariance
    K = Bd * np.diag(y) * Bdt + np.identity(Ny)*np.std(y)**2
    W = np.zeros((Ny, Ny)) #= K.T  Weighting matrix
    BtWB = Gt * Bst * np.diag(1/y) * Bs * G
    β = 0.2 # regularizer strength
    R = R2 # regularizer

    # initialization
    η = Bt * W * B * np.ones_like(y)
    γ = tigre.Ax( np.ones_like(y), geo, angles)
    BtWy = Bt*W*y

    M = [0]

    for p in range(iters):
        proctime = time.process_time()
        print("start iteration:", p)
        for m in M:
            l = tigre.Ax(μ, geo, angles)
            x = np.exp(-l)
            ηox = np.convolve(η, x)
            p = Bt*W*B*x - BtWy - ηox
            L = M*tigre.Atb(( - np.convolve(ηox, x) - np.convolve(p, x) ), geo, angles)
            c = 2*η*p
            lg0 = l>0
            c[lg0] = (2*0.5*η[lg0]+p[lg0]-0.5*η[lg0]*x[lg0]*x[lg0]-x[lg0]*p[lg0]-l[lg0]*(η[lg0]*x[lg0]*x[lg0] + p[lg0]*x[lg0])) / (l[lg0]*l[lg0])
            c[c<0] = 0
            D = M * tigre.Atb(np.convolve(γ, c), geo, angles)
            Δμ = (L+dφ) / (D + ddφ)

            μ = μ - Δμ
            μ[μ<0] = 0

        print("iteration finished: ", time.process_time()-proctime, "s change:", np.mean(Δμ))
    
    return μ


def p_norm(x, p, δ=1):
    δarea = x < δ
    return np.sum(x[δarea])*math.pow(2*δ, -p)*math.pow(δ*δ*2, p-1) + \
           np.sum( np.abs(x[~δarea] - δ*(1-0.5*p) * np.sign(x[~δarea]) ) ** p ) /p

def mlem3(proj, geo, angles, iters, initial=None): # stayman 2013

    μ = tigre.algorithms.fdk(proj, geo, angles).flatten()
    λ = 0 # registration parameters (3 angles, 3 translations)
    H = σI

    Ψp = 1 # rigid transformation
    Ψr = 1 # rigid transformation

    


    y = proj.flatten()
    N = y.shape

    A = lambda x: tigre.Ax(x, geo, angles)

    for n in range(iters):
        for r in range(1, R):
            H[r] = BFGS()
            Δθ = (ψp*δW(λ)*μp).T * δf(ψp*(μ-W(λ)*μp))
            eφ = λ[r-1] + φ * H[r] * ΔΘ(λ[r-1])
            λ[r] = λ[r-1] + eφ * H[r] * ΔΘ(λ[r-1])
        
        λ[0] = λ[-1]
        H[0] = H[-1]

        μ = μ + (
            A(δh*A(μ)) - βr * Ψr * δfr * (Ψr*μ)  - βp* Ψp * δfp * Ψp*(μ - W(λ[-1])*μp )
        ) / (
            A( c * A(μ)) + βr * Ψr**2 * ωfr * (ψr*μ) + βp * ψp**2 * ωfp * Ψp*(μ - W(λ[-1])*μp )
        )
    
    return μ

def CCA(proj, out_shape, geo, angles, iters, use_astra=True): # fessler 1995

    b = 100
    y = b*np.exp(-proj)
    if use_astra:
        μ = utils.FDK_astra(out_shape, geo)(proj, free_memory=False)
    else:
        μ = tigre.algorithms.fdk(proj, geo, angles)
    μ = np.ones(geo.nVoxel, dtype=np.float32)
    r = 0.1
    ω = 0.6
    β = 5

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
        Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)
        Σi_a_ij2 = lambda x: tigre.At2b(x, geo, angles)

    for i in range(iters):
        l = Σj_a_ij( μ )
        ȳ = b*np.exp(-l) + r
        #L̇ = Σi_a_ij( (1-y/ȳ) * b * np.exp(-l) )
        L̇ = Σi_a_ij( ȳ - r - y + r*y/ȳ )

        #L̈ = - Σi_a_ij2( (1-y*r/(ȳ*ȳ)) * b * np.exp(-l) )
        L̈ = - Σi_a_ij2( ȳ - r - y*r/ȳ + y*r*r/(ȳ*ȳ) )

        Ṗ = μm(μ, δψ)
        P̈ = μm(μ, δδψ)

        nom = (L̇ - β * Ṗ)
        den = (-L̈ + β * P̈)

        print(i, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
        print(i, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )

        μ = μ + ω * nom / den

    return μ

def ML_OSTR(proj, out_shape, geo, angles, iters, b=100, use_astra=True):
    
    y = b*np.exp(-proj)
    μ = np.ones(geo.nVoxel, dtype=np.float32)
    r = 0.1

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)

    d = Σi_a_ij( Σj_a_ij(np.ones_like(μ)) * (y-r)**2 / y)
    M = 1 # subsets
    for it in range(iters):
        l̂ = Σj_a_ij(μ)
        ḣ = (y / ( b*np.exp(-l̂) + r ) - 1)*b*np.exp(-l̂)
        n = M * Σi_a_ij(ḣ)
        if it%100 == 0:
            print(it, np.mean(n), np.mean(d), np.median(n), np.median(d), np.mean(n/d), np.median(n/d))
            print(it, np.mean(μ), np.median(μ), np.mean(μ*(n/d)), np.median(μ*(n/d)) )
            
        μ = μ - (n / d)

        μ[μ<0] = 0
    
    return μ

def PL_OSTR(proj, out_shape, geo, angles, iters, use_astra=True):

    if use_astra:
        μ = utils.FDK_astra(out_shape, geo)(proj, free_memory=True)
    else:
        μ = tigre.algorithms.fdk(proj, geo, angles) # initial guess
    
    μ = np.ones(geo.nVoxel, dtype=np.float32)

    b = 100 # i0
    y = b * np.exp(-proj)
    r = 0.1
    β = 5

    M = 1 # subsets

    if use_astra:
        Σj_a_ij = utils.Ax_astra(out_shape, geo)
        Σi_a_ij = utils.Atb_astra(out_shape, geo)
    else:
        Σj_a_ij = lambda x: tigre.Ax(x, geo, angles)
        Σi_a_ij = lambda x: tigre.Atb(x, geo, angles)

    d = Σi_a_ij( Σj_a_ij(np.ones_like(μ)) * (y-r)**2 / y)

    for n in range(iters):
        for m in range(M):
            l̂ = Σj_a_ij(μ)
            ḣ = (y / ( b*np.exp(-l̂) + r ))*b*np.exp(-l̂)
            L̇ = M * Σi_a_ij(ḣ)

            nom = (L̇ + β * μm(μ, δψ) )
            den = (d + 2*β* μm(μ, δδψ) )
            #if iter%10 == 0:
            print(n, np.mean(nom), np.mean(den), np.median(nom), np.median(den), np.mean(nom/den), np.median(nom/den))
            print(n, np.mean(μ), np.median(μ), np.mean(nom/den), np.median(nom/den) )

            μ = μ - nom / den
            μ[μ<0] = 0

    return μ