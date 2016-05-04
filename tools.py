import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import numpy.polynomial.legendre
inv = scipy.linalg.inv


class DMFT(object):
    def __init__(self):
        pass


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

def make_hyb(bath_v, bath_e, freqs, delta):
    """
    Convert bath couplings and energies
    back to hybridization
    """
    nimp = bath_v.shape[0]
    nbath = bath_e.shape[0]
    nw = len(freqs)
    hyb = np.zeros([nimp, nimp, nw], np.complex128)

    for iw, w in enumerate(freqs):
        for p in range(nimp):
            for q in range(nimp):
                for b in range(nbath):
                    hyb[p,q,iw] += bath_v[p,b] * bath_v[q,b] / (w-bath_e[b]+1j*delta)
    return hyb

def get_scaled_legendre_roots(wl, wh, nw):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [wl, wh]

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs, wts = numpy.polynomial.legendre.leggauss(nw)
    freqs *= (wh - wl) / 2. + (wh + wl) / 2.
    wts *= (wh - wl) / 2.
    
    return freqs, wts

def get_bath(hyb, freqs, wts):
    """
    Convert hybridization function 
    to bath couplings and energies

    Args:
        hyb : (nimp, nimp, nw) ndarray
        freqs : (nw) ndarray
        wts : (nw) ndarray, Gaussian wts at freq pts

    Returns:
        bath_v : (nimp, nimp*nw) ndarray
        bath_e : (nimp*nw) ndarray
    """
    nw = len(freqs)
    wh = max(freqs)
    wl = min(freqs)

    dw = (wh - wl) / (nw - 1)
    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb)

    # simple discretization of bath, Eq. (9), arxiv:1507.07468
    v = np.empty_like(v2)

    for iw in range(nw):
        eig, vec = scipy.linalg.eigh(v2[:,:,iw])

        # although eigs should be positive, there
        # could be numerical-zero negative eigs: check this
        neg_eig = [e for e in eig if e < 0]
        assert np.allclose(neg_eig, 0)

        v[:,:,iw] = np.dot(vec, np.diag(np.abs(eig))**0.5) * math.sqrt(wts[iw])

    nimp = hyb.shape[0]
    bath_v = np.reshape(v, [nimp, -1])
    bath_e = np.zeros([nimp * nw])

    # bath_e is [nimp * nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for ip in range(nimp):
        for iw in range(nw):
            bath_e[ip*nw + iw] = freqs[iw]

    return bath_v, bath_e

def get_gf(h, freqs, delta):
    """
    Green's function at a set of frequencies
    """
    n = h.shape[0]
    nw  = len(freqs)
    gf = np.zeros([n, n, nw], np.complex128)
    for iw, w in enumerate(freqs):
        gf[:,:,iw] = inv((w+1j*delta)*np.eye(n)-h)
    return gf

def get_delta(h):
    """
    Rough estimate of broadening from spectrum of h
    """
    n = h.shape[0]
    eigs = scipy.linalg.eigvalsh(h)
    # the factor of 2. is just an empirical estimate
    return 2. * (max(eigs) - min(eigs)) / (n-1.)

def test():
    """
    Main test routine
    """
    n=100 # number of lattice sites
    nimp = 2

    htb=tb(n)

    # frequency range
    wl,wh=-8,8

    # rough estimate of broadening for htb
    delta_lat = get_delta(htb) 
    print "Lattice energy resolution", delta_lat
    
    # number of fitting frequencies
    nw = 40

    # linear freqency
    freqs = np.linspace(wl, wh, nw) 
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)

    # Legendre frequency points. Comment this out
    # to use Legendre polynomial weights
    #freqs, wts = get_scaled_legendre_roots(wl, wh, nw)

    # Hybridization and bath fitting is currently done using delta_lat
    # Lattice GF
    full_gf= get_gf(htb, freqs, delta_lat)

    # Hybridization, and fit bath couplings
    imp_gf0 = get_gf(htb[:nimp,:nimp], freqs, delta_lat)
    hyb = np.zeros_like(imp_gf0)

    for iw in range(len(freqs)):
        hyb[:,:,iw] = inv(imp_gf0[:,:,iw]) - inv(full_gf[:nimp,:nimp,iw])

    bath_v, bath_e = get_bath(hyb, freqs, wts)

    # Setup impurity Hamiltonian including bath couplings
    # and diagonal bath energies
    nbath = len(bath_e)
    himp = np.zeros([nimp + nbath, nimp + nbath])
    himp[:nimp,:nimp] = htb[:nimp,:nimp]
    himp[:nimp, nimp:] = bath_v
    himp[nimp:, :nimp] = bath_v.T
    himp[nimp:,nimp:] = np.diag(bath_e)

    # In principle can use a different broadening when
    # working with himp; here use the same one 
    delta_imp = delta_lat
    print "Impurity energy resolution", delta_imp
    
    # Recompute lattice and impurity Green's function on dense grid
    # for plotting
    dense_freqs = np.linspace(wl, wh, 200)
    imp_gf2 = get_gf(himp, dense_freqs, delta_imp) # here use impurity broadening
    imp_dos2 = -1./np.pi * np.imag(imp_gf2[0,0,:])

    full_gf = get_gf(htb, dense_freqs, delta_lat)
    imp_dos = -1./np.pi * np.imag(full_gf[0,0,:])
    
    plt.plot(dense_freqs, imp_dos2)
    plt.plot(dense_freqs, imp_dos)
    plt.show()


def kernel(dmft):
    """
    DMFT driver
    """
    pass
    
    
def get_hf_gf(himp, eri):
    pass
