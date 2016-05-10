import math
import numpy as np
import numpy.polynomial.legendre
import scipy
import scipy.linalg
inv = scipy.linalg.inv

import pyscf
import pyscf.gto as gto
#import pyscf.scf as scf
import scf_mu as scf
import pyscf.cc.ccsd as ccsd
import pyscf.cc.rccsd_eom as rccsd_eom
import pyscf.ao2mo as ao2mo

import greens_function
import numint_

import matplotlib.pyplot as plt

def _get_delta(h):
    """
    Rough estimate of broadening from spectrum of h
    """
    n = h.shape[0]
    eigs = scipy.linalg.eigvalsh(h)
    # the factor of 2. is just an empirical estimate
    return 2. * (max(eigs) - min(eigs)) / (n-1.)

def _tb(n):
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

def _get_scaled_legendre_roots(wl, wh, nw):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [wl, wh]

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs, wts = numpy.polynomial.legendre.leggauss(nw)
    freqs += 1
    freqs *= (wh - wl) / 2.
    freqs += wl
    wts *= (wh - wl) / 2.
    return freqs, wts

def _get_linear_freqs(wl, wh, nw):
    freqs = np.linspace(wl, wh, nw) 
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)
    return freqs, wts



class DMFT(object):
    """
    DMFT calculation object
    """
    def __init__(self):
        # fill in later
        pass

def kernel(dmft, hcore_kpts, eri, freqs, wts, delta, conv_tol):
    """
    DMFT self-consistency

    Modeled after PySCF HF kernel
    """
    dmft_conv = False
    cycle = 0

    nkpts, nao, nao = hcore_kpts.shape
    hcore_cell = 1./nkpts * np.sum(hcore_kpts, axis=0)

    # get initial guess
    nw = len(freqs)
    sigma = np.zeros([nao, nao, nw])
    hyb = get_hyb(hcore_kpts, sigma, freqs, delta)
    
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb
        bath_v, bath_e = get_bath(hyb, freqs, wts)
        sigma = get_sigma(freqs, hcore_cell, eri, bath_v, \
                          bath_e, delta, dmft.mu)
        hyb = get_hyb(hcore_kpts, sigma, freqs, delta)
        norm_hyb = np.linalg.norm(hyb-hyb_last)

        # this would be a good place to put DIIS
        # or damping
        
        if (norm_hyb < conv_tol):
            dmft_conv = True

        cycle +=1

    return hyb, sigma
    
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

def get_sigma(freqs, hcore_cell, eri, bath_v, bath_e, delta, mu=0):
    """
    Impurity self energy
    """

    nao = hcore_cell.shape[0]
    nbath = len(bath_e)
    himp = np.zeros([nao+nbath, nao+nbath])
    himp[:nao,:nao] = hcore_cell
    himp[:nao,nao:] = bath_v
    himp[nao:,:nao] = bath_v.T
    himp[nao:,nao:] = np.diag(bath_e)

    #gf_imp = get_interacting_gf_ccsd(himp, eri, freqs, \
    #                                 delta, mu)[:nao,:nao,:]
    gf_imp = get_interacting_gf_scf (himp, eri, freqs, \
                                     delta, mu)[:nao,:nao,:]

    nw = len(freqs)
    sigma = np.zeros([nao+nbath,nao+nbath,nw])
    gf0_imp = get_gf(himp, sigma, freqs, delta)[:nao,:nao,:]
 
    sigma = np.zeros_like(gf_imp)
    for iw in range(nw):
        sigma[:,:,iw] = inv(gf0_imp[:,:,iw]) - inv(gf_imp[:,:,iw])
    return sigma
    

def get_interacting_gf_scf (himp, eri_imp, freqs, delta, mu=0):
    n = himp.shape[0]
    nimp = eri_imp.shape[0]
    
    mol = gto.M()
    mol.build()
    #mol.nelectron = n # only half-filling

    mf = scf.RHF(mol, mu)
    # mf.verbose = 4
    mf.max_memory = 1000
    mf.mo_energy = np.zeros([n])
    mf.mo_energy[:n/2] = mf.mu-0.01
    mf.mo_energy[n/2:] = mf.mu+0.01
    
    eri = np.zeros([n,n,n,n])
    eri[:nimp,:nimp,:nimp,:nimp] = eri_imp

    mf.get_hcore = lambda *args: himp
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, eri, n)
    mf.init_guess = '1e'  # currently needed

    mf.scf()
    print 'MF energy = %20.12f\n' % (mf.e_tot)
    print 'MO energies :\n'
    print mf.mo_energy
    print '----\n'

    nw = len(freqs)
    gf = np.zeros([n, n, nw], np.complex128)

    for iw, w in enumerate(freqs):
        resolvent = np.diag(1./((w+1j*delta) * \
                            np.ones([n],np.complex128) - mf.mo_energy))
        gf[:,:,iw] = np.dot(mf.mo_coeff, np.dot(resolvent, \
                                                mf.mo_coeff.T))
    return gf

def get_interacting_gf_ccsd (himp, eri_imp, freqs, delta, mu=0):
    # follow MF code below to get the MF solution
    n = himp.shape[0]
    nimp = eri_imp.shape[0]

    CISD = False
    mol = gto.M()
    mol.build()
    mol.incore_anyway = True
    # needed so that CCSD ao2mo runs properly
    #mol.nelectron = n # only half-filling

    mf = scf.RHF(mol, mu)
    # mf.verbose = 4
    mf.max_memory = 1000
    mf.mo_energy = np.zeros([n])
    mf.mo_energy[:n/2] = mf.mu-0.01
    mf.mo_energy[n/2:] = mf.mu+0.01
    
    eri = np.zeros([n,n,n,n])
    eri[:nimp,:nimp,:nimp,:nimp] = eri_imp

    mf.get_hcore = lambda *args: himp
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, eri, n)
    mf.init_guess = '1e'  # currently needed

    mf.scf()
    print 'MF energy = %20.12f' % (mf.e_tot)
    print 'MO energies :'
    print mf.mo_energy
    print '----\n'

    print "Solving CCSD equations..."
    cc = ccsd.CCSD(mf)
    ecc = cc.ccsd()[0]
    print "CCSD corr = %20.12f" % (ecc)

    print "Solving lambda equations..."
    cc.solve_lambda()

    print "Repeating with EOM CCSD"
    cc_eom = rccsd_eom.RCCSD(mf)

    def ao2mofn_ (mol, bas, compact):
        return ao2mo.incore.general(mf._eri, bas, compact=compact)

    eri_eom = rccsd_eom._ERIS(cc_eom, ao2mofn=ao2mofn_)
    ecc_eom = cc_eom.ccsd(eris=eri_eom)[0]
    print "EOM-CCSD corr = %20.12f" % (ecc_eom)

    #cc_eom.t1 = cc.t1
    #cc_eom.t2 = cc.t2
    cc_eom.l1 = cc.l1
    cc_eom.l2 = cc.l2

    if CISD == True:
        cc_eom.t1 *= 1e-5
        cc_eom.t2 *= 1e-5
        cc_eom.l1 *= 1e-5
        cc_eom.l2 *= 1e-5

    nw = len(freqs)
    gip = np.zeros((n,n,nw), np.complex128)
    gea = np.zeros((n,n,nw), np.complex128)
    gf = greens_function.greens_function()
    # Calculate full (p,q) GF matrix in MO basis
    gip, gea = gf.solve_gf(cc_eom, range(n), range(n), freqs, delta)

    # Change basis from MO to AO
    gip_ao = np.zeros([n, n, nw], np.complex128)
    gea_ao = np.zeros([n, n, nw], np.complex128)
    for iw in range(nw):
        gip_ao[:,:,iw] = np.dot(mf.mo_coeff, np.dot(gip[:,:,iw], \
                                                mf.mo_coeff.T))
        gea_ao[:,:,iw] = np.dot(mf.mo_coeff, np.dot(gea[:,:,iw], \
                                                mf.mo_coeff.T))
    return gip_ao.conj()+gea_ao


def get_hyb(hcore_kpts, sigma, freqs, delta):
    """
    Hybridization
    """
    nw = len(freqs)
    nkpts, nao, nao = hcore_kpts.shape

    gf_kpts = get_gf_kpts(hcore_kpts, sigma, freqs, delta)
    gf_cell = 1./nkpts * np.sum(gf_kpts, axis=0)

    hcore_cell = 1./nkpts * np.sum(hcore_kpts, axis=0)
    gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)

    hyb = np.zeros_like(gf0_cell)
    for iw in range(nw):
        hyb[:,:,iw] = inv(gf0_cell[:,:,iw]) - inv(gf_cell[:,:,iw])
    return hyb


def get_gf_kpts(hcore_kpts, sigma, freqs, delta):
    """
    kpt Green's function at a set of frequencies

    Args: 
         hcore_kpts : (nkpts, nao, nao) ndarray
         sigma : (nao, nao) ndarray
         freqs : (nw) ndarray
         delta : float

    Returns:
         gf_kpts : (nkpts, nao, nao) ndarray
    """
    nw = len(freqs)
    nkpts, nao, nao = hcore_kpts.shape
    gf_kpts = np.zeros([nkpts, nao, nao, nw], np.complex128)

    for k in range(nkpts):
        gf_kpts[k,:,:,:] = get_gf(hcore_kpts[k,:,:], sigma, freqs, delta)
    return gf_kpts
    
def get_gf(hcore, sigma, freqs, delta):
    """
    Green's function at a set of frequencies

    Args: 
         hcore : (nao, nao) ndarray
         sigma : (nao, nao) ndarray
         freqs : (nw) ndarray
         delta : float

    Returns:
         gf : (nao, nao) ndarray

    """
    nw  = len(freqs)
    nao = hcore.shape[0]
    gf = np.zeros([nao, nao, nw], np.complex128)
    for iw, w in enumerate(freqs):
        gf[:,:,iw] = inv((w+1j*delta)*np.eye(nao)-hcore-sigma[:,:,iw])
    return gf

def test():
    nao = 1
    nlat = 32
    U = 4.
    mu = U/2

    htb = _tb(nlat)
    eigs = scipy.linalg.eigvalsh(htb)
    htb_k = np.reshape(eigs, [nlat,nao,nao])
    eri = np.zeros([nao,nao,nao,nao])
    eri[0,0,0,0] = U
    htb_cell = 1./nlat * np.sum(htb_k, axis=0)

    dmft = DMFT()
    dmft.max_cycle = 10
    dmft.mu = mu
    wl, wh = -6, 6
    nw = 31
    delta = _get_delta(htb)
    conv_tol = 1.e-6
    freqs, wts = _get_linear_freqs(wl, wh, nw)
    
    hyb, sigma = kernel(dmft, htb_k, eri, freqs, wts, delta, conv_tol)
    bath_v, bath_e = get_bath(hyb, freqs, wts)

    # conv_gf = get_gf_kpts(htb_k, sigma, freqs, delta)
    # init_gf = get_gf_kpts(htb_k, np.zeros_like(sigma), freqs, delta)
    # conv_imp_gf = 1./nlat * np.sum(conv_gf, axis=0)
    # init_imp_gf = 1./nlat * np.sum(init_gf, axis=0)

    # conv_imp_dos = -1./np.pi * np.imag(np.reshape(conv_imp_gf, [nw]))
    # init_imp_dos = -1./np.pi * np.imag(np.reshape(init_imp_gf, [nw]))
    
    # plt.plot(freqs, conv_imp_dos)
    # plt.plot(freqs, init_imp_dos)
    # plt.show()

    sigma_ = sigma[:,:,0:1].real
    #sigma_ = np.zeros([nao,nao,1])

    def _eval(w, delta):
        #sigma_ = _eval_sigma(w, delta)
        gf_kpts = get_gf_kpts(htb_k, sigma_, [w], delta)
        gf_cell = 1./nlat * np.sum(gf_kpts, axis=0)[:,:,0]
        return np.trace(gf_cell)

    nint_n = numint_.int_quad_real (_eval, \
                                    epsrel=1.0e-5)
    nint_n = numint_.int_quad_imag (_eval, mu, \
                                    epsrel=1.0e-5)

    # nint_i += numint_.int_quad_imag (_eval_XX, mu, \
    #                                    epsrel=1.0e-5)
   
