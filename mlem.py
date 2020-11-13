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

    W = np.zeros((3,3,3), dtype=np.bool)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                if i != 1 and j != 1 and k != 1:
                    W[i, j, k] = 1.0 / np.sqrt((1-i)*(1-i) + (1-j)*(1-j) + (1-k)*(1-k))
    
    W = W.flatten()
    
    def μm(μ, f):
        t = μ.reshape(μshape)
        def filt(data):
            return np.sum([w*f(data[13]-d) for d,w in zip(data,W)])
        return scipy.ndimage.generic_filter(t, filt, size=3).flatten()

    δ = 0.001
    ψ = lambda x: x**2/2 if x <= δ else δ*np.abs(x)-0.5*δ**2
    δψ = lambda x: x if x <= δ else δ*x/np.abs(x)
    δδψ = lambda x: 1 if x <= δ else 0

    for i in range(iters):
        proctime = time.process_time()
        print("start iter ", i)
        l = tigre.Ax(μ.reshape(μshape),geo,angles,'interpolated').flatten()
        print("projected lines: ", time.process_time()-proctime, "s mean value:", np.mean(l), np.median(l), l.shape)
        proctime = time.process_time()

        nom = tigre.Atb((b*np.exp(-l) - y).reshape(yshape), geo, angles).flatten() - β * μm(μ, δψ)
        print("calculated nominator: ", time.process_time()-proctime, "s mean value:", np.mean(nom), np.median(nom), nom.shape)
        proctime = time.process_time()
        denom = tigre.Atb((l*b*np.exp(-l)).reshape(yshape), geo, angles).flatten() + μ*β* μm(μ, δδψ)
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
