import astra
import numpy as np
import utils
import time

def piccs(proj, out_shape, geo, angles, iters, prior_image, real_image=None, use_astra=True):

    α = 0.5
    λ = 10000
    y = proj[:]

    Σj_a_ij = utils.Ax_astra(out_shape, geo)
    Σi_a_ij = utils.Atb_astra(out_shape, geo)
    Σi_a_ij2 = utils.At2b_astra(out_shape, geo)
    I = utils.FDK_astra(out_shape, geo)(proj)

    prior_y = Σj_a_ij(prior_image)
    norm_prior_y = np.linalg.norm(prior_y)**2 + 1e-12
    tv_norm_prior = utils.tv_norm(prior_image) + 1e-12
    δtv_norm_prior = utils.δtv_norm(prior_image)

    proctime = time.perf_counter()

    variation = 1
    #newGrad = -Gradient_fUC(I, prior_image, angles, y, λ, α)
    l = Σj_a_ij(I)
    BPData = Σi_a_ij(l-y)
    Gradient_fPICCS = α * utils.δtv_norm(I-prior_image) + (1-α)*δtv_norm_prior
    newGrad = - 1/tv_norm_prior * Gradient_fPICCS + λ/norm_prior_y*BPData
    tv_norm_I = utils.tv_norm(I)
    fPICCS = α * tv_norm_I + (1-α)*tv_norm_I
    f0 = f1 = fPICCS / tv_norm_prior + 0.5*λ * (l-y).flatten().dot((l-y).flatten()) / norm_prior_y**2
    stepDir = newGrad

    error = []
    for n in range(iters):
        l = Σj_a_ij(I)
        if n > 0:
            # following steps with nonlinear conjugate gradient
            f0 = f1
            gradient = newGrad
            BPData = Σi_a_ij(l-y)
            Gradient_fPICCS = α * utils.δtv_norm(I-prior_image) + (1-α)*δtv_norm_prior
            newGrad = - 1/tv_norm_prior * Gradient_fPICCS + λ/norm_prior_y*BPData
            if n%20 == 0:
                I[I<0] = 0 # non-negative constraint
                stepDir = newGrad
            else:
                beta = np.max(newGrad.flatten().dot((newGrad-gradient).flatten())/(gradient.flatten().dot(gradient.flatten())),0) # Polak-Ribi�re formula
                stepDir = beta*stepDir - gradient # new and old step directions are conjugate (A-orthogonal)

        #eta = Backtracking_Linesearch(I,prior_image,stepDir,newGrad,angles,y,λ,α)            
        c1 = 1e-4
        c2 = 0.5 # following deifinitions in Lauzier(2012)
        eta = 1 # initial step size

        newProjData = Σj_a_ij(I)
        projSearchDir = Σj_a_ij(stepDir)

        s = newProjData - y
        q = projSearchDir

        rho = stepDir.flatten().dot(newGrad.flatten())
        #f0 = fUC(I,priI,redTheta,projData,lambda,alpha)

        tv_norm_I = utils.tv_norm(I)
        fPICCS = α * tv_norm_I + (1-α)*tv_norm_I
        f0 = fPICCS / tv_norm_prior + 0.5*λ * (l-y).flatten().dot((l-y).flatten()) / norm_prior_y

        fRHS = f0 + c1*eta*rho
        lI = I[:]
        lI = lI + eta*stepDir
        #fLHS = fUC(I,priI,redTheta,projData,lambda,alpha)
        
        tv_norm_I = utils.tv_norm(lI)
        fPICCS = α * tv_norm_I + (1-α)*tv_norm_I
        fLHS = fPICCS / tv_norm_prior + 0.5*λ * (l-y).flatten().dot((l-y).flatten()) / norm_prior_y

        #print(fLHS, fRHS, c1*eta*rho, np.abs(fLHS-fRHS))
        while fLHS > fRHS and np.abs(fLHS-fRHS) > 1e-2:
            eta = c2*eta
            lI = I + eta*stepDir
            #fLHS = fast_fUC(I,s,q,eta,priI,redTheta,lambda,alpha)
            tv_norm_I = utils.tv_norm(lI)
            fPICCS = α * tv_norm_I + (1-α)*tv_norm_I
            fLHS = fPICCS / tv_norm_prior + 0.5*λ * np.linalg.norm(s+eta*q)**2 / norm_prior_y
            #ll = Σj_a_ij(lI)
            #fLHS = fPICCS / tv_norm_prior + 0.5*λ * (ll-y).flatten().dot((ll-y).flatten()) / norm_prior_y
            fRHS = f0 + c1*eta*rho
            #print(eta, fLHS, fRHS, np.abs(fLHS-fRHS))

        #print(eta, np.mean(eta*stepDir), np.median(eta*stepDir))
        I = I + eta*stepDir
        #I = reshape(colI,M,N)
        #f1 = fUC(I,prior_image, angles,y,λ,α)
        tv_norm_I = utils.tv_norm(I)
        fPICCS = α * tv_norm_I + (1-α)*tv_norm_I
        f1 = fPICCS / tv_norm_prior + 0.5*λ * (l-y).flatten().dot((l-y).flatten()) / norm_prior_y
        avgVariation = np.abs(f1-f0) + variation
        variation = np.abs(f1-f0)
        #print(n, variation, avgVariation)

        error.append((time.perf_counter()-proctime, np.sum(np.abs(real_image-I))) )
        yield I

    yield I
    yield error