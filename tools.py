import numpy as np
import scipy
import scipy.linalg

def get_linear_freqs(wl, wh, nw):
    freqs = np.linspace(wl, wh, nw) 
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)
    return freqs, wts

def tb(n):
    """
    Tight-binding Hamiltonian
    """
    h=np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            if abs(i-j)==1:
                h[i,j]=1.
    h[0,-1]=1.
    h[-1,0]=1.
    return h

def linear_prediction(x, p, nfit):
    """
    Linear prediction following notation in: 
    https://arxiv.org/pdf/0901.2342.pdf

    x: ndarray of complex data
    p: number of fit coefficients
    nfit: number of points to fit over (reasonable choice is len(x)/2)

    returns: ndarray a, which can be used in the 
    extrapolation formula
    x_n = -\sum_{i=0}^p a_i x_{n-i-1}
    """
    nvalues = len(x)
    R = np.zeros([p,p], np.complex128)
    r = np.zeros([p], np.complex128)
    a = np.zeros([p], np.complex128)
    for i in range(p):
        for j in range(p):
            for n in range(nvalues-nfit,nvalues):
                R[j,i] += np.conj(x[n-j-1]) * x[n-i-1]

    for j in range(p):
        for n in range(nvalues-nfit,nvalues):
            r[j] += np.conj(x[n-j-1])*x[n]

    a = - np.dot(scipy.linalg.pinv(R), r)

    return a

def generate_prediction(a, x, ntotal):
    """
    extrapolation formula
    x_n = -\sum_{i=0}^{p-1} a_i x_{n-i-1}
    """
    nobs = len(x)
    assert ntotal > nobs
    predicted_x = np.zeros([ntotal], np.complex128)
    predicted_x[:nobs] = x
    p = len(a)
    for n in range(nobs, ntotal):        
        for i in range(p):
            predicted_x[n] -= a[i] * predicted_x[n-i-1]

    return predicted_x

def generate_prediction2(a, x, ntotal):
    """
    extrapolation formula
    x_n = -\sum_{i=0}^{p-1} a_i x_{n-i-1}
    """
    nobs = len(x)
    assert ntotal > nobs
    predicted_x = np.zeros([ntotal], np.complex128)
    predicted_x[:nobs] = x

    p=len(a)
    history_x = np.zeros([p], np.complex128)
    for i in range(p):
        history_x[i] = predicted_x[nobs-i-1]

    matA = np.zeros([p, p], np.complex128)
    matA[0]=-a
    matA[1:p,0:p-1]=np.eye(p-1)

    eig, rv = scipy.linalg.eig(matA)
    lv = scipy.linalg.inv(rv)

    # project out growing eigenvectors
    for i, e in enumerate(eig):
        if abs(e) > 1.:
            eig[i] = 0.
            
    pmatA = np.dot(rv, np.dot(np.diag(eig), lv))

    for n in range(nobs, ntotal):        
        history_x = np.dot(pmatA, history_x)
        predicted_x[n] = history_x[0]
    return predicted_x


def predict_gf(gf, ntotal):
    """
    GF prediction, following
    https://arxiv.org/pdf/0901.2342.pdf

    currently predict for each element separately, could try for all elements together.

    ntotal: total number of time points

    returns: predicted GF
    """
    predicted_gf = np.zeros([gf.shape[0], gf.shape[1], ntotal],
                            dtype=gf.dtype)
    nobs = gf.shape[2]
    predicted_gf[:,:,:nobs] = gf

    nfit = nobs/2
    for p in range(gf.shape[0]):
        for q in range(gf.shape[1]):
            
            a = linear_prediction(gf[p,q,:], nfit, nfit)
            predicted_x = generate_prediction2(a, gf[p,q,:], ntotal)
            predicted_gf[p,q,:] = predicted_x
    return predicted_gf

def get_gfw(gft, times, freqs, delta):
    """
    frequency transform of GF with Gaussian broadening delta
    """
    ti,tf=times[0],times[-1]
    gfw = np.zeros([gft.shape[0],gft.shape[1],len(freqs)], np.complex128)
    for iw, w in enumerate(freqs):
        ftwts = 1./(tf-ti) * np.exp(1j*w*times) * np.exp(-.5*delta**2*(times**2))
        for p in range(gft.shape[0]):
            for q in range(gft.shape[1]):
                gfw[p,q,iw] = scipy.integrate.romb(ftwts * gft[p,q])

    return gfw
