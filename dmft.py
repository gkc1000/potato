#!/usr/bin/python

from sys import stdout

import numpy as np
import numpy.polynomial.legendre
import scipy
import scipy.linalg
import scipy.optimize
inv = scipy.linalg.inv
import h5py

import pyscf
import pyscf.gto as gto
#import pyscf.scf as scf
import scf_mu as scf
import pyscf.cc.ccsd as ccsd
import pyscf.cc.rccsd_eom as rccsd_eom
import pyscf.ao2mo as ao2mo

import greens_function
import numint_

fci_ = False
try:
    import PyCheMPS2
    import ctypes
    fci_ = True
except:
    pass

def _get_delta(eigs):
    """
    Rough estimate of broadening from spectrum of h
    """
    n = eigs.shape[0]
    # the factor of 2. is just an empirical estimate
    return 2. * (max(eigs) - min(eigs)) / (n-1.)

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

def get_sigma (mf_gf, corr_gf):
    nw = mf_gf.shape[2]
    sigma = np.zeros_like(mf_gf)
    for iw in range(nw):
        sigma[:,:,iw] = inv(mf_gf[:,:,iw]) - inv(corr_gf[:,:,iw])
    return sigma

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

class FCIsol:
    def __init__ (self, HamCheMPS2, theFCI, GSvector, GSenergy):
        assert (fci_)

        assert (isinstance(HamCheMPS2, PyCheMPS2.PyHamiltonian))
        self.HamCheMPS2 = HamCheMPS2
        assert (isinstance(theFCI, PyCheMPS2.PyFCI))
        self.FCI = theFCI
        self.GSvector = GSvector
        self.GSenergy = GSenergy

def fci_kernel (mf_):
    norb = mf_.mo_coeff.shape[0]
    h0   = 0.
    h1t  = np.dot(mf_.mo_coeff.T, \
                  np.dot(mf_.get_hcore(), mf_.mo_coeff))
    erit = ao2mo.incore.full(mf_._eri, mf_.mo_coeff, compact=False)
    erit = erit.reshape([norb,norb,norb,norb])

    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    Group = 0
    orbirreps = np.zeros((norb,), dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(norb, Group, orbirreps)
    HamCheMPS2.setEconst( h0 )
    for cnt1 in range(norb):
        for cnt2 in range(norb):
            HamCheMPS2.setTmat(cnt1, cnt2, h1t[cnt1,cnt2])
            for cnt3 in range(norb):
                for cnt4 in range(norb):
                    HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, \
                                       erit[cnt1,cnt3,cnt2,cnt4])

    nel = np.count_nonzero(mf_.mo_occ)*2
    assert( nel % 2 == 0 )
    Nel_up       = nel / 2
    Nel_down     = nel / 2
    Irrep        = 0
    maxMemWorkMB = 100.0
    FCIverbose   = 0
    theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, \
                              Irrep, maxMemWorkMB, FCIverbose )
    GSvector = np.zeros( [ theFCI.getVecLength() ], \
                         dtype=ctypes.c_double )
    GSvector[ theFCI.LowestEnergyDeterminant() ] = 1
    EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
    print "FCI corr = %20.12f" % (EnergyCheMPS2-mf_.e_tot)
    print '====\n'

    fcisol = FCIsol(HamCheMPS2, theFCI, GSvector, EnergyCheMPS2)
    return fcisol

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

def fci_gf (freqs, delta, fcisol, mo_coeff):
    n  = mo_coeff.shape[0]
    nw = len(freqs)
    gf = np.zeros([n, n, nw], np.complex128)

    orbsLeft  = np.arange(n, dtype=ctypes.c_int)
    orbsRight = np.arange(n, dtype=ctypes.c_int)

    theFCI     = fcisol.FCI
    energy_gs  = fcisol.GSenergy
    gs_vector  = fcisol.GSvector
    HamCheMPS2 = fcisol.HamCheMPS2
    for iw, w in enumerate(freqs):
        if np.iscomplex(w):
            wr = w.real
            wi = w.imag
        else:
            wr = w
            wi = 0.
        ReGF, ImGF = theFCI.GFmatrix_rem (wr-energy_gs, 1.0, wi+delta, \
                orbsLeft, orbsRight, 1, gs_vector, HamCheMPS2)
        gf_ = (ReGF.reshape((n,n), order='F') + \
               1j*ImGF.reshape((n,n), order='F')).T

        ReGF, ImGF = theFCI.GFmatrix_add (wr+energy_gs, -1.0, wi+delta, \
                orbsLeft, orbsRight, 1, gs_vector, HamCheMPS2)
        gf_ += ReGF.reshape((n,n), order='F') + \
               1j*ImGF.reshape((n,n), order='F')
        gf[:,:,iw] = np.dot(mo_coeff, np.dot(gf_, mo_coeff.T))
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
            delta, conv_tol=1.e-6, dmpf=0.5, chkpt=True):
    """
    DMFT self-consistency

    Modeled after PySCF HF kernel
    """
    dmft_conv = False
    cycle = 0

    nkpts, nao, nao = hcore_kpts.shape
    hcore_cell = 1./nkpts * np.sum(hcore_kpts, axis=0)
    if np.iscomplexobj(hcore_cell):
        assert (np.allclose(np.zeros((nao,nao,)), hcore_cell.imag))
        hcore_cell = hcore_cell.real

    # get initial guess
    nw = len(freqs)

    if dmft.sigma is None:
        sigma = np.zeros([nao, nao, nw])
    else:
        assert (dmft.sigma.shape == (nao,nao,nw))
        sigma = dmft.sigma.copy()
    gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)
    gf_cell = np.zeros([nao, nao, nw], np.complex128)
    for k in range(nkpts):
        gf_cell += 1./nkpts * \
                   get_gf(hcore_kpts[k,:,:], sigma, freqs, delta)
    hyb = get_sigma(gf0_cell, gf_cell)

    dmft.delta = delta
    dmft.freqs = freqs
    dmft.wts   = wts

    def _gf_imp (freqs, delta, mf_, corr_=None):
        if dmft.solver_type != 'scf':
            assert (corr_ is not None)

        if dmft.solver_type == 'scf':
            return mf_gf (freqs, delta, mf_.mo_coeff, mf_.mo_energy)
        elif dmft.solver_type == 'cc':
            return cc_gf (freqs, delta, corr_, mf_.mo_coeff)
        elif dmft.solver_type == 'fci':
            assert (fci_)
            return fci_gf (freqs, delta, corr_, mf_.mo_coeff)
    
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb
        bath_v, bath_e = get_bath(hyb, freqs, wts)
        himp, eri_imp = imp_ham(hcore_cell, eri_cell, bath_v, bath_e)

        dmft.mf_ = mf_kernel (himp, eri_imp, dmft.mu)
        if dmft.solver_type == 'cc':
            dmft.corr_ = cc_kernel (dmft.mf_)
        elif dmft.solver_type == 'fci':
            dmft.corr_ = fci_kernel (dmft.mf_)

        if dmft.solver_type == 'scf':
            gf_imp = _gf_imp (freqs, delta, dmft.mf_)
        elif dmft.solver_type in ('cc', 'fci'):
            gf_imp = _gf_imp (freqs, delta, dmft.mf_, dmft.corr_)
        gf_imp = gf_imp[:nao,:nao,:]

        nb = bath_e.shape[0]
        sgdum = np.zeros((nb+nao,nb+nao,nw))
        gf0_imp = get_gf(himp, sgdum, freqs, delta)
        gf0_imp = gf0_imp[:nao,:nao,:]

        sigma = get_sigma(gf0_imp, gf_imp)

        gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)
        gf_cell = np.zeros([nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf_cell += 1./nkpts * \
                       get_gf(hcore_kpts[k,:,:], sigma, freqs, delta)
        hyb_new = get_sigma(gf0_cell, gf_cell)

        # damping
        hyb = dmpf*hyb_new + (1-dmpf)*hyb

        dmft.hyb   = hyb
        dmft.sigma = sigma
        if chkpt:
            dmft.chkpt()

        norm_hyb = np.linalg.norm(hyb-hyb_last)
        print 'cycle    = ', cycle+1
        print 'norm_hyb = ', norm_hyb
        print '****'
        stdout.flush()
        
        if (norm_hyb < conv_tol):
            dmft_conv = True
        cycle +=1
    dmft.conv_   = dmft_conv

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

class DMFT:
    def __init__(self, hcore_k, eri_cell, \
                 max_cycle=32, solver_type='scf'):
        self.nkpts, self.nao = hcore_k.shape[:2]
        assert (hcore_k.shape == (self.nkpts, self.nao, self.nao,))
        assert (eri_cell.shape == (self.nao,)*4)

        self.hcore_k  = hcore_k
        self.eri_cell = eri_cell

        self.max_cycle   = max_cycle
        self.solver_type = solver_type
        self.chkfile     = None

        # do not touch
        self.mu    = None
        self.mf_   = None
        self.corr_ = None

        self.conv_   = False
        self.hyb     = None
        self.sigma   = None
        self.freqs   = None
        self.wts     = None

    def chkpt (self):
        if self.chkfile is not None:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['dmft/hyb']   = self.hyb
                fh5['dmft/sigma'] = self.sigma

                fh5['dmft/solver_type'] = self.solver_type
                fh5['dmft/mu']          = self.mu
                fh5['dmft/delta']       = self.delta
                fh5['dmft/freqs']       = self.freqs
                fh5['dmft/wts']         = self.wts
                fh5['dmft/hcore_k']     = self.hcore_k
                fh5['dmft/eri_cell']    = self.eri_cell

    def kernel (self, mu, freqs, wts, delta, \
                conv_tol=1.e-6, dmpf=0.5):
        self.mu = mu
        kernel (self, self.hcore_k, self.eri_cell, \
                freqs, wts, delta, conv_tol, dmpf)

    def kernel_nopt (self, n0, mu0, freqs, wts, delta, \
                     conv_tol=1.e-6, dmpf=0.5, tol=1.e-4):
        def n_eval (mu):
            self.kernel (mu, freqs, wts, delta, \
                         conv_tol, dmpf)
            n_ = self.n_int (delta, epsrel=0.1*tol)
            print 'mu = ', mu
            print 'nint_n [imag] = ', n_
            return n0-n_

        mu = mu0
        mu = scipy.optimize.newton (n_eval, mu, tol=tol)
        self.mu = mu

    def _gf (self, freqs, delta):
        assert (self.conv_)
        if self.solver_type != 'scf':
            assert (self.corr_ is not None)
        if self.solver_type == 'scf':
            gf = mf_gf (freqs, delta, \
                        self.mf_.mo_coeff, self.mf_.mo_energy)
        elif self.solver_type == 'cc':
            gf = cc_gf (freqs, delta, self.corr_, self.mf_.mo_coeff)
        elif self.solver_type == 'fci':
            assert (fci_)
            gf = fci_gf (freqs, delta, self.corr_, self.mf_.mo_coeff)
        return gf[:self.nao,:self.nao,:]

    def _gf0 (self, freqs, delta):
        himp = self.mf_.get_hcore()
        nb = himp.shape[0]
        nw = len(freqs)
        sig_dum = np.zeros((nb,nb,nw,))
        gf = get_gf(himp, sig_dum, freqs, delta)
        return gf[:self.nao,:self.nao,:]

    def _local_sigma (self, freqs, delta):
        gf0_ = self._gf0 (freqs, delta)
        gf1_ = self._gf (freqs, delta)
        return get_sigma(gf0_, gf1_)

    def get_ldos (self, freqs, delta, sigma=None):
        nw = len(freqs)
        nao = self.nao
        nkpts = self.nkpts

        if sigma is None:
            sigma = self._local_sigma (freqs, delta)
        gf = np.zeros([nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf += 1./nkpts * get_gf(self.hcore_k[k,:,:], sigma, \
                                    freqs, delta)
        dos = np.zeros([nao, nw])
        for k in range(nao):
           dos[k,:] += -1./np.pi * np.imag(gf[k,k,:])
        return dos

    def get_ldos_ni (self, freqs, delta):
        nw = len(freqs)
        nao = self.nao
        sigma = np.zeros([nao, nao, nw])
        return self.get_ldos (freqs, delta, sigma)

    def rdm_freq (self, freqs, delta):
        nao = self.nao
        nkpts = self.nkpts

        if isinstance(freqs, (float, np.float, complex, np.complex)):
            _freqs = np.array([freqs])
        else:
            _freqs = freqs

        nw = len(_freqs)
        sigma = self._local_sigma (_freqs, delta)
        p = np.zeros([nao, nao, nw], np.complex128)
        for k in range(nkpts):
            p += 1./nkpts * get_gf(self.hcore_k[k,:,:], sigma, \
                                   _freqs, delta)
        return p

    def rdm_int (self, delta, axis='imag', \
                 x0=None, epsrel=1.0e-4):
        assert (axis in ('real', 'imag'))
        if axis == 'real':
            assert (x0 is not None)

        p_int = np.zeros([self.nao, self.nao])
        for k in range(self.nao):
            for l in range(self.nao):
                def _p_freq (w, delta):
                    return self.rdm_freq (w, delta)[k,l,0]

                if axis == 'imag':
                    p_int[k,l] = numint_.int_quad_imag (_p_freq, \
                                self.mu, epsrel=epsrel, delta=delta)
                else:
                    assert (x0 is not None)
                    p_int[k,l] = numint_.int_quad_real (_p_freq, \
                                self.mu, x0=x0, \
                                epsrel=epsrel, delta=delta)
        if axis == 'imag':
            return 2*0.5*(np.eye(self.nao)-p_int)
        else:
            return 2*p_int

    def n_freq (self, freqs, delta):
        if isinstance(freqs, (float, np.float, complex, np.complex)):
            _freqs = np.array([freqs])
        else:
            _freqs = freqs

        nw = len(_freqs)
        n = np.zeros([nw], np.complex128)
        for iw, w in enumerate(_freqs):
            n[iw] = np.trace(self.rdm_freq(np.array([w]), \
                                           delta)[:,:,0])
        return n

    def n_int (self, delta, axis='imag', \
               x0=None, epsrel=1.0e-4):
        assert (axis in ('real', 'imag'))
        if axis == 'real':
            assert (x0 is not None)

        def _n_freq (w, delta):
            return self.n_freq (w, delta)[0]

        if axis == 'imag':
            # NL = # poles left of mu, NR = # poles right of mu
            # nao = NL + NR
            # integration gives NR - NL (factor of 2 in imag_fn)
            nint_n = numint_.int_quad_imag (_n_freq, self.mu, \
                                epsrel=epsrel, delta=delta)
            return 2*0.5*(self.nao-nint_n)
        else:
            nint_n = numint_.int_quad_real (_n_freq, self.mu, \
                                x0=x0, epsrel=epsrel, delta=delta)
            return 2*nint_n

    def energy (self, delta, axis='imag', \
                x0=None, epsrel=1.0e-4):
        assert (axis in ('real', 'imag'))
        if axis == 'real':
            assert (x0 is not None)

        nkpts = self.nkpts
        inf_  = np.array([100000.])
        if axis == 'imag':
            sinf = self._local_sigma(1j*inf_+self.mu, delta)[:,:,0]
        else:
            sinf = self._local_sigma(inf_, delta)[:,:,0]

        def _eval_en0 (w, delta):
            sigma = self._local_sigma (np.array([w]), delta)
            en = np.complex(0.)
            for k in range(nkpts):
                gf_ = get_gf(self.hcore_k[k,:,:], sigma, \
                             np.array([w]), delta)[:,:,0]
                en += 1./nkpts * \
                      np.trace(np.dot(self.hcore_k[k,:,:], gf_))
            return en
        def _eval_en1(w, delta):
            sigma = self._local_sigma (np.array([w]), delta)
            en = np.complex(0.)
            for k in range(nkpts):
                gf_ = get_gf(self.hcore_k[k,:,:], sigma, \
                             np.array([w]), delta)[:,:,0]
                en += 1./nkpts * \
                      np.trace(np.dot(sinf, gf_))
            return en
        def _eval_en2(w, delta):
            sigma = self._local_sigma (np.array([w]), delta)
            en = np.complex(0.)
            for k in range(nkpts):
                gf_ = get_gf(self.hcore_k[k,:,:], sigma, \
                             np.array([w]), delta)[:,:,0]
                en += 1./nkpts * \
                      np.trace(np.dot(sigma[:,:,0]-sinf, gf_))
            return en

        if axis == 'imag':
            # trace of h with GF
            nint_e0 = numint_.int_quad_imag (_eval_en0, self.mu, \
                                    epsrel=epsrel, delta=delta)
            print 'nint H_c    [imag] = ', -nint_e0

            # energy due to 1/w self-energy
            nint_e2 = numint_.int_quad_imag (_eval_en2, self.mu, \
                                    epsrel=epsrel, delta=delta)
            print 'nint S[w]   [imag] = ', -nint_e2/2.

            # energy due to a constant self-energy
            nint_e1 = numint_.int_quad_imag (_eval_en1, self.mu, \
                                    epsrel=epsrel, delta=delta)
            e1 = (np.real(np.trace(sinf)) - nint_e1)
            print 'nint S[inf] [imag] = ', e1/2
            return -nint_e0 + e1/2. -nint_e2/2.

        else:
            # trace of h with GF
            nint_e0 = numint_.int_quad_real (_eval_en0, self.mu, \
                                    x0=x0, epsrel=epsrel, delta=delta)
            print 'nint H_c    [real] = ', 2*nint_e0

            # energy due to 1/w self-energy
            nint_e2 = numint_.int_quad_real (_eval_en2, self.mu, \
                                    x0=x0, epsrel=epsrel, delta=delta)
            print 'nint S[w]   [real] = ', nint_e2

            # energy due to a constant self-energy
            nint_e1 = numint_.int_quad_real (_eval_en1, self.mu, \
                                    x0=x0, epsrel=epsrel, delta=delta)
            print 'nint S[inf] [real] = ', nint_e1
            return 2*nint_e0 + nint_e1 + nint_e2


def hub_1d (nx, U, nw, fill=1., chkf=None, \
            max_cycle=256, solver_type='scf'):
    kx = np.arange(-nx/2+1, nx/2+1, dtype=float)
    hcore_k_ = -2*np.cos(2.*np.pi*kx/nx)
    hcore_k  = hcore_k_.reshape([nx,1,1])
    eri = np.zeros([1,1,1,1])
    eri[0,0,0,0] = U
    mu0 = U/2.
    # print np.sort(hcore_k_)
    # assert(False)

    dmft = DMFT (hcore_k, eri, \
                 max_cycle=max_cycle, solver_type=solver_type)
    dmft.chkfile = chkf

    wl, wh = -5.+U/2., 5.+U/2.
    delta = _get_delta(hcore_k_)
    #freqs, wts = _get_linear_freqs(wl, wh, nw)
    freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    dmft.kernel_nopt (fill, mu0, freqs, wts, delta, dmpf=0.75)
    return dmft, freqs, delta

def hub_2d (nx, ny, U, nw, fill=1., chkf=None, \
            max_cycle=256, solver_type='scf'):
    kx = np.arange(-nx/2+1, nx/2+1, dtype=float)
    ky = np.arange(-ny/2+1, ny/2+1, dtype=float)
    kx_, ky_ = np.meshgrid(kx,ky)
    hcore_k_ = -2*np.cos(2.*np.pi*kx_.flatten(order='C')/nx) \
               -2*np.cos(2.*np.pi*ky_.flatten(order='C')/ny)
    hcore_k  = hcore_k_.reshape([nx*ny,1,1])
    eri = np.zeros([1,1,1,1])
    eri[0,0,0,0] = U
    mu0 = U/2.
    # print np.sort(hcore_k_)
    # assert(False)

    dmft = DMFT (hcore_k, eri, \
                 max_cycle=max_cycle, solver_type=solver_type)
    dmft.chkfile = chkf

    wl, wh = -7.+U/2., +7.+U/2.
    delta = _get_delta(hcore_k_)
    #freqs, wts = _get_linear_freqs(wl, wh, nw)
    freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    dmft.kernel_nopt (fill, mu0, freqs, wts, delta, dmpf=0.75)
    return dmft, freqs, delta

def hub_cell_1d (nx, isx, U, nw, fill=1., chkf=None, \
                 max_cycle=256, solver_type='scf'):
    assert (nx % isx == 0)

    def nn_hopping (ns):
        t = np.zeros((ns,ns,), dtype=float)
        for ist in range(ns-1):
            t[ist,ist+1] = -1.0
            t[ist+1,ist] = -1.0
        t[0,-1] += -1.0
        t[-1,0] += -1.0
        return t

    def planewave (ns):
        U = np.zeros((ns,ns,), dtype=complex)
        scr = np.arange(ns, dtype=float)
        for k in range(ns):
            kk = (2.0*np.pi/ns)*k
            U[:,k] = np.exp(1j*kk*scr)
        U *= (1.0/np.sqrt(ns))
        return U

    nx_ = nx/isx
    T   = nn_hopping (nx)
    Ut  = planewave (nx_)
    hcore_k = np.zeros((nx_,isx,isx,), dtype=complex)
    for i1 in range(isx):
        for i2 in range(isx):
            for k in range(nx_):
                T_ = T[i1::isx,i2::isx].\
                     reshape((nx_,nx_,), order='F')
                hcore_k[k,i1,i2] = \
                        np.dot(Ut[:,k].T, np.dot(T_, Ut[:,k].conj()))

    hcore_k_ = np.zeros((nx_,isx))
    for k in range(nx_):
        hcore_k_[k,:] = scipy.linalg.eigh(hcore_k[k,:,:], \
                                          eigvals_only=True)
    # print np.sort(hcore_k_.flatten())
    # assert (False)

    eri = np.zeros([isx,isx,isx,isx])
    for k in range(isx):
        eri[k,k,k,k] = U
    mu0 = U/2.

    dmft = DMFT (hcore_k, eri, \
                 max_cycle=max_cycle, solver_type=solver_type)
    dmft.chkfile = chkf

    wl, wh = -5.+U/2., 5.+U/2.
    delta = _get_delta(hcore_k_.flatten())
    #freqs, wts = _get_linear_freqs(wl, wh, nw)
    freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    dmft.kernel_nopt (fill*isx, mu0, freqs, wts, delta, dmpf=0.75)
    return dmft, freqs, delta

def hub_cell_2d (nx, ny, isx, isy, U, nw, fill=1., chkf=None, \
                 max_cycle=256, solver_type='scf'):
    assert (nx % isx == 0)
    assert (ny % isy == 0)

    def nn_hopping (nx, ny):
        ns = nx*ny
        t = np.zeros((ny,nx,ny,nx,), dtype=float)
        for istx in range(nx):
            for isty in range(ny-1):
                t[isty,istx,isty+1,istx] = -1.0
                t[isty+1,istx,isty,istx] = -1.0
            t[0,istx,-1,istx] += -1.0
            t[-1,istx,0,istx] += -1.0
        for isty in range(ny):
            for istx in range(nx-1):
                t[isty,istx,isty,istx+1] = -1.0
                t[isty,istx+1,isty,istx] = -1.0
            t[isty,0,isty,-1] += -1.0
            t[isty,-1,isty,0] += -1.0
        return t

    def planewave (nx, ny):
        ns = nx*ny
        U = np.zeros((ns,ns,), dtype=complex)
        sx = np.arange(nx, dtype=float)
        sy = np.arange(ny, dtype=float)
        scry, scrx = np.meshgrid(sx,sy)
        scrx = scrx.reshape((ns,), order='F')
        scry = scry.reshape((ns,), order='F')
        k = 0
        for kx in range(nx):
            kkx = (2.0*np.pi/nx)*kx
            for ky in range(ny):
                kky = (2.0*np.pi/ny)*ky
                U[:,k] = np.exp(1j*(kkx*scrx+kky*scry))
                k += 1
        U *= (1.0/np.sqrt(ns))
        return U

    nx_ = nx/isx
    ny_ = ny/isy
    ns_ = nx_*ny_
    T   = nn_hopping (nx, ny)
    Ut  = planewave (nx_, ny_)

    hcore_k = np.zeros((ns_,isy,isx,isy,isx,), dtype=complex)
    for i1y in range(isy):
        for i1x in range(isx):
            for i2y in range(isy):
                for i2x in range(isx):
                    T_ = T[i1y::isy,i1x::isx,i2y::isy,i2x::isx].\
                         reshape((ns_,ns_,), order='F')
                    for k in range(ns_):
                        hcore_k[k,i1y,i1x,i2y,i2x] = \
                            np.dot(Ut[:,k].T, np.dot(T_, Ut[:,k].conj()))
    hcore_k = hcore_k.reshape((ns_,isy*isx,isy*isx,), order='F')

    hcore_k_ = np.zeros((ns_,isx*isy))
    for k in range(ns_):
        hcore_k_[k,:] = scipy.linalg.eigh(hcore_k[k,:,:], \
                                          eigvals_only=True)
    # print np.sort(hcore_k_.flatten())
    # assert (False)

    eri = np.zeros([isx*isy,isx*isy,isx*isy,isx*isy])
    for k in range(isx*isy):
        eri[k,k,k,k] = U
    mu0 = U/2.

    dmft = DMFT (hcore_k, eri, \
                 max_cycle=max_cycle, solver_type=solver_type)
    dmft.chkfile = chkf

    wl, wh = -7.+U/2., 7.+U/2.
    delta = _get_delta(hcore_k_.flatten())
    #freqs, wts = _get_linear_freqs(wl, wh, nw)
    freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    dmft.kernel_nopt (fill*isx*isy, mu0, freqs, wts, delta, dmpf=0.75)
    return dmft, freqs, delta


if __name__ == '__main__':
    U = 4.
    # dmft, w, delta = hub_1d (30, U, 15, solver_type='scf')
    # dmft, w, delta = hub_2d (10, 10, U, 9, solver_type='scf')

    # dmft, w, delta = hub_cell_1d (30, 2, U, 9, \
    #                               solver_type='scf')
    # dmft, w, delta = hub_cell_2d (10, 10, 2, 2, U, 7, \
    #                               solver_type='scf')

    try:
        import matplotlib.pyplot as plt
        freqs = _get_linear_freqs(-5., 5., 128)[0]
        plt.figure(1)
        dos0 = dmft.get_ldos_ni (freqs, delta)
        plt.plot(freqs, dos0[0,:])

        freqs = _get_linear_freqs(-5.+U/2., 5.+U/2., 128)[0]
        plt.figure(2)
        dos1 = dmft.get_ldos (freqs, delta)
        plt.plot(freqs, dos1[0,:])
        plt.show()
    except:
        pass

    lr = False

    print '\nnumber of electrons'
    n_ = dmft.n_int (delta)
    print 'nint_n [imag] = ', n_
    if lr:
        nr_ = dmft.n_int (delta, \
                          axis='real', x0=-20.)
        print 'nint_n [real] = ', nr_
    print '----\n'

    # print 'density matrix'
    # p_ = dmft.rdm_int (delta)
    # print 'nint_p [imag] = '
    # print p_
    # if lr:
    #     pr_ = dmft.rdm_int (delta, \
    #                         axis='real', x0=-20.)
    #     print 'nint_p [real] = '
    #     print pr_
    # print '----\n'

    print 'energy'
    e_ = dmft.energy (delta)
    print 'nint_e [imag] = ', e_
    if lr:
        er_ = dmft.energy (delta, \
                           axis='real', x0=-20.)
        print 'nint_e [real] = ', er_
    print '----\n'


