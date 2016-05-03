import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
inv = scipy.linalg.inv

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

def hyb_to_bath(hyb, freqs):
    """
    Convert hybridization function 
    to bath couplings and energies

    Args:
        hyb : (nimp, nimp, nw) ndarray
        freqs : (nw) ndarray

    Returns:

        bath_v : (nimp, nimp*nw) ndarray
        bath_e : (nimp*nw) ndarray
    """
    nw = len(freqs)
    wh = max(freqs)
    wl = min(freqs)

    dw = (wh - wl) / (nw - 1)
    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb) * dw

    # simple discretization of bath, Eq. (9), arxiv:1507.07468
    v = np.empty_like(v2)
    
    for iw in range(nw):
        eig, vec = scipy.linalg.eigh(v2[:,:,iw])

        # although eigs should be positive, there
        # could be numerical-zero negative eigs: check this
        neg_eig = [e for e in eig if e < 0]
        assert np.allclose(neg_eig, 0)

        v[:,:,iw] = np.dot(vec, np.diag(np.abs(eig))**0.5)

    nimp = hyb.shape[0]
    bath_v = np.reshape(v, [nimp, -1])
    bath_e = np.zeros([nimp * nw])

    # bath_e is [nimp * nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for ip in range(nimp):
        for iw in range(nw):
            bath_e[ip*nw + iw] = freqs[iw]
        
    return bath_v, bath_e

def test():
    """
    Main test routine
    """
    # n: number of sites
    n=100
    nimp = 4

    htb=tb(n)

    # frequency range
    wl,wh=-8,8
    delta = .5 * (wh-wl) / n
    freqs = np.arange(wl,wh,0.1)
    nw = len(freqs)

    # full lattice GF
    full_gf=np.zeros([n,n,nw], np.complex128)

    # Compute reference DOS
    for iw, w in enumerate(freqs):
        full_gf[:,:,iw]=inv((w+1j*delta)*np.eye(n)-htb)

    exact_dos= -1./np.pi * 1./n * np.imag(np.einsum("iiw->w", full_gf))
    imp_dos = -1./np.pi * np.imag(full_gf[0,0,:])

    # Compute hybridization, and fit bath couplings
    imp_gf0 = np.zeros([nimp,nimp,nw], np.complex128)
    hyb = np.zeros([nimp,nimp,nw], np.complex128)
    for iw, w in enumerate(freqs):
        imp_gf0[:,:,iw] = inv((w + 1j*delta) * np.eye(nimp) - htb[:nimp,:nimp])
        hyb[:,:,iw] = inv(imp_gf0[:,:,iw]) - inv(full_gf[:nimp,:nimp,iw])

    bath_v, bath_e = hyb_to_bath(hyb, freqs)

    # # Check hybridization
    # hyb2 = make_hyb(bath_v, bath_e, freqs, delta)
    # for iw, w in enumerate(freqs):
    #     print hyb[0,0, iw], hyb2[0,0,iw]


    # Setup impurity Hamiltonian including bath couplings
    # and diagonal bath energies
    nbath = len(bath_e)
    himp = np.zeros([nimp + nbath, nimp + nbath])
    himp[:nimp,:nimp] = htb[:nimp,:nimp]
    himp[:nimp, nimp:] = bath_v
    himp[nimp:, :nimp] = bath_v.T
    himp[nimp:,nimp:] = np.diag(bath_e)

    # Compute impurity Green's function
    imp_gf2 = np.zeros([nimp+nbath,nimp+nbath,nw], np.complex128)
    for iw, w in enumerate(freqs):
        imp_gf2[:,:,iw]=inv((w+1j*delta)*np.eye(nimp+nbath)-himp)

    imp_dos2 = -1./np.pi * np.imag(imp_gf2[0,0,:])

    plt.plot(freqs, imp_dos2)
    plt.plot(freqs, imp_dos)
    plt.show()
