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

def get_sigma(mf_gf, corr_gf):
    nw = mf_gf.shape[2]
    sigma = np.zeros_like(mf_gf)
    for iw in range(nw):
        sigma[:,:,iw] = inv(mf_gf[:,:,iw]) - inv(corr_gf[:,:,iw])
    return sigma

def get_hyb (gf0_cell, gf_cell):
    nw = gf0_cell.shape[2]
    hyb = np.zeros_like(gf0_cell)
    for iw in range(nw):
        hyb[:,:,iw] = inv(gf0_cell[:,:,iw]) - inv(gf_cell[:,:,iw])
    return hyb

def mf_kernel (himp, eri_imp, mu):
    n = himp.shape[0]
    mol = gto.M()
    mol.build()
    #mol.nelectron = n # only half-filling

    mf = scf.RHF(mol, mu)
    mf.verbose = 0
    # mf.verbose = 4
    mf.max_memory = 1000
    mf.mo_energy = np.zeros([n])
    mf.mo_energy[:n/2] = mf.mu-0.01
    mf.mo_energy[n/2:] = mf.mu+0.01

    mf.get_hcore = lambda *args: himp
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, eri_imp, n)
    mf.init_guess = '1e'  # currently needed

    _ = mf.scf()
    print 'MF energy = %20.12f\n' % (mf.e_tot)
    print 'MO energies :\n'
    print mf.mo_energy
    print '----\n'
    return mf

def cc_kernel (mf_):
    CISD = False

    print "Solving CCSD equations..."
    cc = ccsd.CCSD(mf_)
    ecc = cc.ccsd()[0]
    print "CCSD corr = %20.12f" % (ecc)

    print "Solving lambda equations..."
    cc.solve_lambda()

    print "Solving EOM CCSD"
    cc_eom = rccsd_eom.RCCSD(mf_)

    def ao2mofn_ (mol, bas, compact):
        return ao2mo.incore.general(mf_._eri, bas, compact=compact)

    eri_eom = rccsd_eom._ERIS(cc_eom, ao2mofn=ao2mofn_)
    ecc_eom = cc_eom.ccsd(eris=eri_eom)[0]
    print "EOM-CCSD corr = %20.12f" % (ecc_eom)
    print '====\n'

    #cc_eom.t1 = cc.t1
    #cc_eom.t2 = cc.t2
    cc_eom.l1 = cc.l1
    cc_eom.l2 = cc.l2

    if CISD == True:
        cc_eom.t1 *= 1e-5
        cc_eom.t2 *= 1e-5
        cc_eom.l1 *= 1e-5
        cc_eom.l2 *= 1e-5
    return cc_eom

def mf_gf (freqs, delta, mo_coeff, mo_energy):
    nw = len(freqs)
    n = mo_coeff.shape[0]
    gf = np.zeros([n, n, nw], np.complex128)
    for iw, w in enumerate(freqs):
        g = np.diag(1./((w+1j*delta) * \
                        np.ones([n], np.complex128) - mo_energy))
        gf[:,:,iw] = np.dot(mo_coeff, np.dot(g, mo_coeff.T))
    return gf

def cc_gf (freqs, delta, cc_eom, mo_coeff):
    n = mo_coeff.shape[0]
    nw = len(freqs)
    gip = np.zeros((n,n,nw), np.complex128)
    gea = np.zeros((n,n,nw), np.complex128)
    gf = greens_function.greens_function()
    # Calculate full (p,q) GF matrix in MO basis
    g_ip = gf.solve_ip(cc_eom, range(n), range(n), \
                       freqs.conj(), delta).conj()
    g_ea = gf.solve_ea(cc_eom, range(n), range(n), \
                       freqs, delta)

    # Change basis from MO to AO
    gf = np.zeros([n, n, nw], np.complex128)
    for iw, w in enumerate(freqs):
        g_ip_ = np.dot(mo_coeff, np.dot(g_ip[:,:,iw], mo_coeff.T))
        g_ea_ = np.dot(mo_coeff, np.dot(g_ea[:,:,iw], mo_coeff.T))
        gf[:,:,iw] = g_ip_+g_ea_
    return gf

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

def imp_ham (hcore_cell, eri_cell, bath_v, bath_e):
    nao = hcore_cell.shape[0]
    nbath = len(bath_e)
    himp = np.zeros([nao+nbath, nao+nbath])
    himp[:nao,:nao] = hcore_cell
    himp[:nao,nao:] = bath_v
    himp[nao:,:nao] = bath_v.T
    himp[nao:,nao:] = np.diag(bath_e)

    eri_imp = np.zeros([nao+nbath, nao+nbath, nao+nbath, nao+nbath])
    eri_imp[:nao,:nao,:nao,:nao] = eri_cell
    return himp, eri_imp

def kernel (dmft, hcore_kpts, eri_cell, freqs, wts, \
            delta, conv_tol):
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
    gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)
    gf_cell = np.zeros([nao, nao, nw], np.complex128)
    for k in range(nkpts):
        gf_cell += 1./nkpts * \
                   get_gf(hcore_kpts[k,:,:], sigma, freqs, delta)
    hyb = get_hyb(gf0_cell, gf_cell)

    def _gf_imp (freqs, delta, mf_, corr_=None):
        if dmft.solver_type == 'scf':
            return mf_gf (freqs, delta, mf_.mo_coeff, mf_.mo_energy)
        elif dmft.solver_type == 'cc':
            assert (corr_ is not None)
            return cc_gf (freqs, delta, corr_, mf_.mo_coeff)
    
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb
        bath_v, bath_e = get_bath(hyb, freqs, wts)
        himp, eri_imp = imp_ham(hcore_cell, eri_cell, bath_v, bath_e)

        dmft.mf_ = mf_kernel (himp, eri_imp, dmft.mu)
        if dmft.solver_type == 'cc':
            dmft.corr_ = cc_kernel (dmft.mf_)

        if dmft.solver_type == 'scf':
            gf_imp = _gf_imp (freqs, delta, dmft.mf_)
        elif dmft.solver_type == 'cc':
            gf_imp = _gf_imp (freqs, delta, dmft.mf_, dmft.corr_)
        gf_imp = gf_imp[:nao,:nao]

        nb = bath_e.shape[0]
        sgdum = np.zeros((nb+nao,nb+nao,nw))
        gf0_imp = get_gf(himp, sgdum, freqs, delta)[:nao,:nao]

        sigma = get_sigma(gf0_imp, gf_imp)

        gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)
        gf_cell = np.zeros([nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf_cell += 1./nkpts * \
                       get_gf(hcore_kpts[k,:,:], sigma, freqs, delta)
        hyb = get_hyb(gf0_cell, gf_cell)

        norm_hyb = np.linalg.norm(hyb-hyb_last)

        # this would be a good place to put DIIS
        # or damping
        
        if (norm_hyb < conv_tol):
            dmft_conv = True
        cycle +=1

    dmft.hyb     = hyb
    dmft.freqs   = freqs
    dmft.wts     = wts
    dmft.himp    = himp
    dmft.eri_imp = eri_imp
    
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

        v[:,:,iw] = np.dot(vec, np.diag(np.sqrt(np.abs(eig)))) * \
                    np.sqrt(wts[iw])

    nimp = hyb.shape[0]
    bath_v = np.reshape(v, [nimp, -1])
    bath_e = np.zeros([nimp * nw])

    # bath_e is [nimp * nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for ip in range(nimp):
        for iw in range(nw):
            bath_e[ip*nw + iw] = freqs[iw]

    return bath_v, bath_e

class DMFT(object):
    """
    DMFT calculation object
    """
    def __init__(self):
        # not input options
        self.mf_   = None
        self.corr_ = None


def test():
    nao = 1
    nlat = 64
    U = 4.
    mu = U/2

    htb = _tb(nlat)
    eigs = scipy.linalg.eigvalsh(htb)
    htb_k = np.reshape(eigs, [nlat,nao,nao])
    eri = np.zeros([nao,nao,nao,nao])
    eri[0,0,0,0] = U

    dmft = DMFT()
    dmft.max_cycle = 32
    dmft.mu = mu
    dmft.solver_type = 'scf'

    wl, wh = -6, +6
    nw = 63
    delta = _get_delta(htb)
    conv_tol = 1.e-6
    freqs, wts = _get_linear_freqs(wl, wh, nw)

    kernel (dmft, htb_k, eri, freqs, wts, delta, conv_tol)

    def _gf_imp (w, delta, mf_, corr_=None):
        if dmft.solver_type == 'scf':
            return mf_gf (w, delta, mf_.mo_coeff, mf_.mo_energy)
        elif dmft.solver_type == 'cc':
            assert (corr_ is not None)
            return cc_gf (w, delta, corr_, mf_.mo_coeff)

    def _gf (w, delta):
        if dmft.solver_type == 'scf':
            gf_imp = _gf_imp (w, delta, dmft.mf_)
        elif dmft.solver_type == 'cc':
            gf_imp = _gf_imp (w, delta, dmft.mf_, dmft.corr_)
        return gf_imp[:nao,:nao]
    def _gf0 (w, delta):
        nb = dmft.himp.shape[0]-nao
        nw = len(w)
        sig_dum = np.zeros((nb+nao,nb+nao,nw))
        return get_gf(dmft.himp, sig_dum, w, delta)[:nao,:nao]

    def _local_sigma (w, delta):
        gf0_ = _gf0 (w, delta)
        gf1_ = _gf (w, delta)
        return get_sigma(gf0_, gf1_)

    def _local_gf (hcore, sigma, w, delta):
        return get_gf(hcore, sigma, w, delta)

    nkpts  = nlat
    delta_ = delta
    freqs_ = _get_linear_freqs(-10., 10., 256)[0]

    ldos = True   # whether to plot local dos
    lplt = False  # whether to plot functions to integrate
    lr   = True   # whether to carry out integrations on the real axis

    if ldos:
        sigma = _local_sigma (freqs_, delta_)
        sgdum = np.zeros_like(sigma)
        gf0 = np.zeros([nao, nao, freqs_.shape[0]], np.complex128)
        gf1 = np.zeros([nao, nao, freqs_.shape[0]], np.complex128)
        for k in range(nkpts):
            gf0 += 1./nkpts * _local_gf(htb_k[k,:,:], sgdum, \
                                        freqs_, delta_)
            gf1 += 1./nkpts * _local_gf(htb_k[k,:,:], sigma, \
                                        freqs_, delta_)

        dos0 = np.zeros([freqs_.shape[0]])
        dos1 = np.zeros([freqs_.shape[0]])
        for k in range(nao):
           dos0[:] += -1./np.pi * np.imag(gf0[k,k,:])
           dos1[:] += -1./np.pi * np.imag(gf1[k,k,:])

        plt.plot(freqs_, dos0)  # non-interacting DOS
        plt.plot(freqs_, dos1)
        plt.show()

    def _eval_p(w, delta):
        sigma = _local_sigma (np.array([w]), delta)
        p = np.zeros([nao, nao], np.complex128)
        for k in range(nkpts):
            p += 1./nkpts * _local_gf(htb_k[k,:,:], sigma, \
                                      np.array([w]), delta)[:,:,0]
        return p
    def _eval_n(w, delta):
        return np.trace(_eval_p(w, delta))

    sigma_infi = _local_sigma(np.array([1j*1000000.+mu]), delta)[:,:,0]
    sigma_infr = _local_sigma(np.array([1000000.]), delta)[:,:,0]

    def _eval_en0(w, delta):
        sigma = _local_sigma (np.array([w]), delta)
        en = np.complex(0.)
        for k in range(nkpts):
            gf_ = _local_gf(htb_k[k,:,:], sigma, \
                            np.array([w]), delta)[:,:,0]
            en += 1./nkpts * np.trace(np.dot(htb_k[k,:,:], gf_))
        return en
    def _eval_en1(w, delta):
        sigma = _local_sigma (np.array([w]), delta)
        en = np.complex(0.)
        for k in range(nkpts):
            gf_ = _local_gf(htb_k[k,:,:], sigma, \
                            np.array([w]), delta)[:,:,0]
            if np.iscomplex(w):
                en += 1./nkpts * np.trace(np.dot(sigma_infi, gf_))
            else:
                en += 1./nkpts * np.trace(np.dot(sigma_infr, gf_))
        return en
    def _eval_en2(w, delta):
        sigma = _local_sigma (np.array([w]), delta)
        en = np.complex(0.)
        for k in range(nkpts):
            gf_ = _local_gf(htb_k[k,:,:], sigma, \
                            np.array([w]), delta)[:,:,0]
            if np.iscomplex(w):
                en += 1./nkpts * np.trace(np.dot(sigma[:,:,0]\
                                                 -sigma_infi, gf_))
            else:
                en += 1./nkpts * np.trace(np.dot(sigma[:,:,0]\
                                                 -sigma_infr, gf_))
        return en

    if lplt:
        def real_fn(w, gf_fn):
            return -1./np.pi * np.imag(gf_fn(w, delta_))
        def imag_fn(w, gf_fn):
            return -2./np.pi * np.real(gf_fn(1j*w+mu, delta_))

        fnr0 = np.zeros_like(freqs_)
        fnr1 = np.zeros_like(freqs_)
        fnr2 = np.zeros_like(freqs_)
        fnr3 = np.zeros_like(freqs_)
        fni0 = np.zeros_like(freqs_)
        fni1 = np.zeros_like(freqs_)
        fni2 = np.zeros_like(freqs_)
        fni3 = np.zeros_like(freqs_)
        wmin = np.min(freqs_)
        wmax = np.max(freqs_)
        for iw, w in enumerate(freqs_):
            fnr0[iw] = real_fn(w+mu, _eval_n)
            fnr1[iw] = real_fn(w+mu, _eval_en0)
            fnr2[iw] = real_fn(w+mu, _eval_en1)
            fnr3[iw] = real_fn(w+mu, _eval_en2)
            fni0[iw] = imag_fn(w, _eval_n)
            fni1[iw] = imag_fn(w, _eval_en0)
            fni2[iw] = imag_fn(w, _eval_en1)
            fni3[iw] = imag_fn(w, _eval_en2)

        plt.plot(freqs_+mu, fnr0)
        plt.figure()
        plt.plot(freqs_+mu, fnr1)
        plt.figure()
        plt.plot(freqs_+mu, fnr2)
        plt.figure()
        plt.plot(freqs_+mu, fnr3)
        plt.figure()
        plt.plot(freqs_, fni0)
        plt.figure()
        plt.plot(freqs_, fni1)
        plt.figure()
        plt.plot(freqs_, fni2)
        plt.figure()
        plt.plot(freqs_, fni3)
        plt.show()

    # NL = # poles to left of mu, NR = # poles to right of mu
    # nao = NL + NR
    # integration gives NR - NL (factor of 2 in imag_fn)
    print '\nnumber of electrons'
    if True:
        nint_n = numint_.int_quad_imag (_eval_n, mu, \
                                        epsrel=1.0e-4, delta=delta_)
        nint_n = 2*0.5*(nao-nint_n)
        print 'nint_n [imag] = ', nint_n
        # additional factor of 2 by spin integration
    if lr:
        nint_n = numint_.int_quad_real (_eval_n, mu, x0=-40., \
                                        epsrel=1.0e-4, delta=delta_)
        print 'nint_n [real] = ', 2*nint_n
    print '----\n'

    if True:
        print 'energy [imag]'
        # trace of h with GF
        nint_e0 = numint_.int_quad_imag (_eval_en0, mu, \
                                         epsrel=1.0e-4, delta=delta_)
        print 'nint H_c    [imag] = ', -nint_e0

        # energy due to 1/w self-energy
        nint_e2 = numint_.int_quad_imag (_eval_en2, mu, \
                                         epsrel=1.0e-4, delta=delta_)
        print 'nint S[w]   [imag] = ', -nint_e2/2.

        # energy due to a constant self-energy
        nint_e1 = numint_.int_quad_imag (_eval_en1, mu, \
                                         epsrel=1.0e-4, delta=delta_)
        e1 = (np.real(np.trace(sigma_infi)) - nint_e1)
        print 'nint S[inf] [imag] = ', e1/2
        print 'nint_e = ', -nint_e0 + e1/2. -nint_e2/2.
        print '----\n'

    if lr:
        print 'energy [real]'
        nint_e0 = numint_.int_quad_real (_eval_en0, mu, x0=-40., \
                                         epsrel=1.0e-4, delta=delta_)
        print 'nint H_c    [real] = ', 2*nint_e0

        # energy due to 1/w self-energy
        nint_e2 = numint_.int_quad_real (_eval_en2, mu, x0=-40., \
                                         epsrel=1.0e-4, delta=delta_)
        print 'nint S[w]   [real] = ', nint_e2

        # energy due to a constant self-energy
        nint_e1 = numint_.int_quad_real (_eval_en1, mu, x0=-40., \
                                         epsrel=1.0e-4, delta=delta_)
        print 'nint S[inf] [real] = ', nint_e1
        print 'nint_e = ', 2*nint_e0 + nint_e1 + nint_e2
        print '----\n'


