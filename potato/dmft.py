#!/usr/bin/python

import time
import sys

import numpy as np
from scipy import linalg, optimize
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, ao2mo, cc
import scf_mu as scf

import ccgf

fci_ = False
try:
    import PyCheMPS2
    import ctypes
    fci_ = True
except:
    pass


def kernel(dmft, mu0, occupancy=1., delta=None, sigma0=None,
           conv_tol=None, opt_mu=False, mu_tol=1.e-2):
    '''DMFT at fixed occupancy, searching over mu'''
    dmft_conv = False
    if opt_mu:
        def _occupancy_error(mu):
            logger.info(dmft, 'trying mu = %s', mu)
            dmft_conv = kernel_mu(dmft, mu, delta=delta, sigma0=sigma0,
                                  conv_tol=conv_tol)
            occupancy_ = np.trace(dmft.get_rdm_imp())/(dmft.nao)
            logger.info(dmft, 'current occupancy = %s', occupancy_)
            logger.info(dmft, 'target occupancy = %s', occupancy)
            return occupancy-occupancy_

        #TODO: Automate choice of bounds
        mu_min, mu_max = -0.1, 0.1
        mu = optimize.brentq(_occupancy_error, mu_min, mu_max, xtol=mu_tol)
        occupancy_ = np.trace(dmft.get_rdm_imp())/(dmft.nao)
        logger.info(dmft, 'converged on mu = %s', mu)
        logger.info(dmft, 'calculated occupancy = %s', occupancy_)
        logger.info(dmft, 'target occupancy = %s', occupancy)
    else:
        mu = mu0
        dmft_conv = kernel_mu(dmft, mu, delta=delta, sigma0=sigma0,
                              conv_tol=conv_tol)

    return dmft_conv, mu

def kernel_mu(dmft, mu, delta=None, sigma0=None,
              conv_tol=None, dump_chk=True):
    '''DMFT self-consistency cycle at fixed mu'''
    cput0 = (time.clock(), time.time())

    if delta is None:
        delta = dmft.delta
    if conv_tol is None:
        conv_tol = dmft.conv_tol

    hcore_k = dmft.hcore_k
    eris = dmft.eris
    delta = dmft.delta

    nkpts, nao, nao = hcore_k.shape
    hcore_cell = 1./nkpts * np.sum(hcore_k, axis=0)
    if np.iscomplexobj(hcore_cell):
        assert (np.allclose(np.zeros((nao,nao,)), hcore_cell.imag))
        hcore_cell = hcore_cell.real

    nw = dmft.nbath
    #TODO: automate the choice of frequency range, or allow user choice
    #      although the optimal choice may depend on mu, which is changing
    U = eris[0,0,0,0]
    wl, wh = -7.+U/2, 7.+U/2
    #wl, wh = -9.+U/2, 9.+U/2
    #freqs, wts = _get_linear_freqs(wl, wh, nw)
    freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    dmft.freqs = freqs
    dmft.wts = wts

    if sigma0 is None:
        sigma = dmft.get_init_sigma()
    else:
        assert (sigma0.shape == (nao,nao,nw))
        sigma = sigma0.copy()

    gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)
    gf_cell = np.zeros([nao, nao, nw], np.complex128)
    for k in range(nkpts):
        gf_cell += 1./nkpts * get_gf(hcore_k[k,:,:], sigma, freqs, delta)
    hyb = get_sigma(gf0_cell, gf_cell)

    if isinstance(dmft.diis, lib.diis.DIIS):
        dmft_diis = dmft.diis
    elif dmft.diis:
        dmft_diis = lib.diis.DIIS(dmft, dmft.diis_file)
        dmft_diis.space = dmft.diis_space
    else:
        dmft_diis = None
    diis_start_cycle = dmft.diis_start_cycle

    dmft_conv = False
    cycle = 0
    cput1 = logger.timer(dmft, 'initialize DMFT', *cput0)
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb
        bath_v, bath_e = get_bath(hyb, freqs, wts)
        himp, eri_imp = imp_ham(hcore_cell, eris, bath_v, bath_e)

        dmft._scf = mf_kernel(himp, eri_imp, mu, verbose=dmft.verbose)
        gf_imp = dmft.get_gf_imp(freqs, delta)
        gf_imp = gf_imp[:nao,:nao,:]

        nb = bath_e.shape[0]
        sgdum = np.zeros((nb+nao,nb+nao,nw))
        gf0_imp = get_gf(himp, sgdum, freqs, delta)
        gf0_imp = gf0_imp[:nao,:nao,:]

        sigma = get_sigma(gf0_imp, gf_imp)

        gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)
        gf_cell = np.zeros([nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf_cell += 1./nkpts * get_gf(hcore_k[k,:,:], sigma, freqs, delta)
        hyb_new = get_sigma(gf0_cell, gf_cell)

        damp = dmft.damp
        if (abs(damp) > 1e-4 and
            (0 <= cycle < diis_start_cycle-1 or dmft_diis is None)):
            hyb_new = damp*hyb_new + (1-damp)*hyb
        hyb = dmft.run_diis(hyb_new, cycle, dmft_diis)

        dmft.hyb = hyb
        dmft.sigma = sigma

        norm_dhyb = np.linalg.norm(hyb-hyb_last)
        logger.info(dmft, 'cycle= %d  |dhyb|= %4.3g', cycle+1, norm_dhyb)

        if (norm_dhyb < conv_tol):
            dmft_conv = True

        #if dump_chk and dmft.chkfile:
        #    dmft.dump_chk()

        #if dmft.verbose >= logger.DEBUG:
        #    write_dos_and_sigma(dmft, cycle+1)

        cput1 = logger.timer(dmft, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    logger.timer(dmft, 'DMFT_cycle', *cput0)
    return dmft_conv

def write_dos_and_sigma(dmft, cycle):
    cput0 = (time.clock(), time.time())

    U = dmft.eris[0,0,0,0]
    eta = 0.5 # hardcoded :-(
    occupancy = dmft.occupancy
    if occupancy is None:
        occupancy = 1.00

    verbose = dmft._scf.mol.verbose
    dmft._scf.mol.verbose = 0

    wl, wh = -10.+U/2., 10.+U/2.
    freqs = np.linspace(wl, wh, 64)
    ldos = dmft.get_ldos_imp(freqs, eta)[0]

    filename = 'ldos_%d.dat'%(cycle)
    with open(filename, 'w') as f:
        f.write('# mu = %0.12g\n'%(dmft.mu))
        for w,freq in enumerate(freqs):
            f.write('%0.12g %.12g %.12g\n'%(freq, freq-dmft.mu, ldos[w]))

    omega_ns = np.linspace(0., 20., 64)
    sigma = dmft.get_sigma_imp(1j*omega_ns, 0.0)[0,0]
    sigma = np.imag(sigma)
    filename = 'sigma_%d.dat'%(cycle)
    with open(filename, 'w') as f:
        for n,wn in enumerate(omega_ns):
            f.write('%.12g %.12g\n'%(wn, sigma[n]))

    dmft._scf.mol.verbose = verbose
    logger.timer(dmft, 'write LDOS and sigma for debugging', *cput0)

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
    freqs, wts = np.polynomial.legendre.leggauss(nw)
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
        sigma[:,:,iw] = linalg.inv(mf_gf[:,:,iw]) - linalg.inv(corr_gf[:,:,iw])
    return sigma

def mf_kernel(himp, eri_imp, mu, verbose=logger.NOTE):
    n = himp.shape[0]
    mol = gto.M()
    mol.verbose = verbose
    mol.build()

    mf = scf.RHF(mol, mu)
    mf.max_memory = 1000
    mf.mo_energy = np.zeros([n])

    mf.get_hcore = lambda *args: himp
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, eri_imp, n)
    mf.init_guess = '1e'  # currently needed

    mf.smearing = None
    mf.scf()
    if mf.converged:
        return mf
    else:
        # If SCF does not converge, try smearing for convergence, then do
        # one-shot HF without smearing
        mf.smearing = 0.1
        mf.scf()
        if mf.converged is False:
            raise RuntimeError('SCF with smearing not converged.')
        dm0 = mf.make_rdm1()
        mf.smearing = None
        mf.max_cycle = 1
        mf.scf(dm0=dm0)
        return mf


def mf_gf(mf, freqs, delta, ao_orbs=None):
    ''' Calculate the mean-field GF matrix in the AO basis'''
    nmo = mf.mo_coeff.shape[0]
    if ao_orbs is None:
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nw = len(freqs)
    gf = np.zeros([nmo, nmo, nw], np.complex128)
    for iw, w in enumerate(freqs):
        g = np.diag(1./((w+1j*delta) * \
                        np.ones([nmo], np.complex128) - mf.mo_energy))
        gf[:,:,iw] = np.dot(mf.mo_coeff, np.dot(g, mf.mo_coeff.T))

    return gf[:nao,:nao]


def cc_gf(mf, freqs, delta, ao_orbs=None, gmres_tol=1e-4):
    ''' Calculate the CCSD GF matrix in the AO basis'''
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    mycc = cc.RCCSD(mf)
    mycc.max_cycle = 200
    #mycc.diis = False
    mycc.ccsd()
    if mycc.converged is False:
        raise RuntimeError('Ground-state CCSD not converged.')
    mycc.solve_lambda()
    gf = ccgf.CCGF(mycc, tol=gmres_tol)
    # Note .conj()'s to make this the retarded GF
    g_ip = gf.ipccsd_ao(ao_orbs, freqs.conj(), mf.mo_coeff, delta).conj()
    g_ea = gf.eaccsd_ao(ao_orbs, freqs, mf.mo_coeff, delta)
    gf = g_ip + g_ea

    return gf[:nao,:nao]


def cc_rdm(mf, ao_orbs=None):
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    mycc = cc.RCCSD(mf)
    mycc.ccsd()
    mycc.solve_lambda()
    rdm_mo = mycc.make_rdm1()
    rdm = np.dot(mf.mo_coeff, np.dot(rdm_mo, mf.mo_coeff.T))

    return rdm[:nao,:nao]


class FCIsol:
    def __init__(self, HamCheMPS2, theFCI, GSvector, GSenergy):
        assert (fci_)

        assert (isinstance(HamCheMPS2, PyCheMPS2.PyHamiltonian))
        self.HamCheMPS2 = HamCheMPS2
        assert (isinstance(theFCI, PyCheMPS2.PyFCI))
        self.FCI = theFCI
        self.GSvector = GSvector
        self.GSenergy = GSenergy

def fci_kernel(mf):
    norb = mf.mo_coeff.shape[0]
    h0 = 0.
    h1t = np.dot(mf.mo_coeff.T, \
                 np.dot(mf.get_hcore(), mf.mo_coeff))
    erit = ao2mo.incore.full(mf._eri, mf.mo_coeff, compact=False)
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

    nel = np.count_nonzero(mf.mo_occ)*2
    assert( nel % 2 == 0 )
    Nel_up = nel / 2
    Nel_down = nel / 2
    Irrep = 0
    maxMemWorkMB = 100.0
    FCIverbose = 0
    theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, \
                              Irrep, maxMemWorkMB, FCIverbose )
    GSvector = np.zeros( [ theFCI.getVecLength() ], \
                         dtype=ctypes.c_double )
    GSvector[ theFCI.LowestEnergyDeterminant() ] = 1
    EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
    print "FCI corr = %20.12f" % (EnergyCheMPS2-mf.e_tot)

    fcisol = FCIsol(HamCheMPS2, theFCI, GSvector, EnergyCheMPS2)
    return fcisol


def fci_gf(mf, freqs, delta, ao_orbs=None, gmres_tol=1e-4):
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nmo = mf.mo_coeff.shape[0]
    nw = len(freqs)
    gf = np.zeros([nmo, nmo, nw], np.complex128)

    orbsLeft  = np.arange(nmo, dtype=ctypes.c_int)
    orbsRight = np.arange(nmo, dtype=ctypes.c_int)

    fcisol = fci_kernel(mf)
    theFCI = fcisol.FCI
    energy_gs = fcisol.GSenergy
    gs_vector = fcisol.GSvector
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
        gf_ = (ReGF.reshape((nmo,nmo), order='F') + \
               1j*ImGF.reshape((nmo,nmo), order='F')).T

        ReGF, ImGF = theFCI.GFmatrix_add (wr+energy_gs, -1.0, wi+delta, \
                orbsLeft, orbsRight, 1, gs_vector, HamCheMPS2)
        gf_ += ReGF.reshape((nmo,nmo), order='F') + \
               1j*ImGF.reshape((nmo,nmo), order='F')
        gf[:,:,iw] = np.dot(mf.mo_coeff, np.dot(gf_, mf.mo_coeff.T))
    return gf[:nao,:nao]


def fci_rdm(mf, ao_orbs=None):
    if ao_orbs is None:
        nmo = mf.mo_coeff.shape[0]
        ao_orbs = range(nmo)
    nao = len(ao_orbs)
    nmo = mf.mo_coeff.shape[0]
    fcisol = fci_kernel(mf)
    theFCI = fcisol.FCI
    gs_vector = fcisol.GSvector
    rdm2 = np.zeros(nmo**4) 
    theFCI.Fill2RDM(gs_vector, rdm2) 
    rdm2 = rdm2.reshape((nmo,nmo,nmo,nmo))
    rdm_mo = np.einsum('ikkj->ij', rdm2.transpose((0,1,3,2)))/(nmo-1)
    rdm = np.dot(mf.mo_coeff, np.dot(rdm_mo, mf.mo_coeff.T))

    return rdm[:nao,:nao]


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
        gf[:,:,iw] = linalg.inv((w+1j*delta)*np.eye(nao)-hcore-sigma[:,:,iw])
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
        eig, vec = linalg.eigh(v2[:,:,iw])

        # although eigs should be positive, there
        # could be numerical-zero negative eigs: check this
        neg_eigs = [e for e in eig if e < 0]
        if not np.allclose(neg_eigs, 0):
            log = logger.Logger(sys.stdout, 4)
            for neg_eig in neg_eigs:
                log.warn('hyb eval = %.8f', neg_eig)
            raise RuntimeError('hybridization has negative eigenvalues')

        v[:,:,iw] = np.dot(vec, np.diag(np.sqrt(np.abs(eig)))) * \
                    np.sqrt(wts[iw])

    nimp = hyb.shape[0]
    # bath_v[p,k_n] is the coupling btw impurity site p and bath orbital k
    # (total number nw=nbath) belonging to bath n (total number nimp)
    bath_v = v.reshape([nimp, nimp*nw])
    bath_e = np.zeros([nimp*nw])

    # bath_e is [nimp*nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for ip in range(nimp):
        for iw in range(nw):
            bath_e[ip*nw + iw] = freqs[iw]

    return bath_v, bath_e

class DMFT(lib.StreamObject):
    max_cycle = 50
    conv_tol = 1e-3
    damp = 0.7
    gmres_tol = 1e-4

    diis = True
    diis_space = 6
    diis_start_cycle = 1
    diis_file = None
    def __init__(self, hcore_k, eris, nbath, solver_type='scf'):
        self.nkpts, self.nao = hcore_k.shape[:2]
        assert (hcore_k.shape == (self.nkpts, self.nao, self.nao,))
        assert (eris.shape == (self.nao,)*4)

        self.hcore_k = hcore_k
        self.eris = eris
        self.nbath = nbath
        self.solver_type = solver_type

        self.verbose = logger.NOTE
        self.chkfile = None

        # do not touch
        self.stdout = sys.stdout
        self.mu = None
        self._mf = None
        self._corr = None

        self.converged = False
        self.hyb = None
        self.sigma = None
        self.freqs = None
        self.wts = None

    def dump_flags(self):
        if self.verbose < logger.INFO:
            return self

        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'impurity solver = %s', self.solver_type)
        logger.info(self, 'n impurity orbitals = %d', self.nao)
        logger.info(self, 'n bath orbitals per impurity = %d', self.nbath)
        logger.info(self, 'n bath orbitals total = %d', self.nbath*self.nao)
        logger.info(self, 'nkpts in lattice = %d', self.nkpts)
        if self.opt_mu:
            logger.info(self, 'mu will be optimized, init guess = %g, target occupancy = %g',
                        self.mu, self.occupancy)
        else:
            logger.info(self, 'mu is fixed, mu = %g', self.mu)
        logger.info(self, 'damping factor = %g', self.damp)
        logger.info(self, 'DMFT convergence tol = %g', self.conv_tol)
        logger.info(self, 'max. DMFT cycles = %d', self.max_cycle)
        logger.info(self, 'GMRES convergence tol = %g', self.gmres_tol)
        logger.info(self, 'delta for discretization = %g', self.delta)
        logger.info(self, 'using diis = %s', self.diis)
        if self.diis:
            logger.info(self, 'diis_space = %d', self.diis_space)
            logger.info(self, 'diis_start_cycle = %d', self.diis_start_cycle)
        if self.chkfile:
            logger.info(self, 'chkfile to save DMFT result = %s', self.chkfile)
        return self

    def dump_chk(self):
        if self.chkfile:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['dmft/hyb'] = self.hyb
                fh5['dmft/sigma'] = self.sigma
                fh5['dmft/solver_type'] = self.solver_type
                fh5['dmft/mu'] = self.mu
                fh5['dmft/delta'] = self.delta
                fh5['dmft/freqs'] = self.freqs
                fh5['dmft/wts'] = self.wts
                fh5['dmft/hcore_k'] = self.hcore_k
                fh5['dmft/eris'] = self.eris
        return self


    def get_init_sigma(self, freqs=None):
        nao, nbath = self.nao, self.nbath
        sigma = np.empty([nao,nao,nbath], dtype=np.complex)
        U = self.eris[0,0,0,0]
        if freqs is None:
            freqs = self.freqs
        if self.occupancy is None:
            occupancy = 1.
        else:
            occupancy = self.occupancy
        for iw,w in enumerate(freqs):
            sigma[:,:,iw] = U*occupancy/2.*np.eye(nao)
        return sigma


    def kernel(self, mu0, occupancy=None, delta=0.1, sigma0=None,
               conv_tol=None, opt_mu=False, mu_tol=1.e-2):
        '''main routine for DMFT

        Args:
            mu0 : float
                Chemical potential or an initial guess if opt_mu=True
        
        Kwargs:
            occupancy : None or float
                Target average occupancy (1 is half filling)
            delta : float
                Broadening used during self-consistency
            sigma0 : numpy ndarray
                Initial guess for the self-energy
            conv_tol : float
                Convergence tolerance on the hybridization
            opt_mu : bool
                Whether to optimize the chemical potential
            mu_tol : float
                Convergence tolerance on the optimization of the chemical
                potential
        '''

        cput0 = (time.clock(), time.time())
        self.mu = mu0
        self.occupancy = occupancy
        self.delta = delta
        if conv_tol:
            self.conv_tol = conv_tol
        self.opt_mu = opt_mu

        self.dump_flags()

        self.converged, self.mu = kernel(self, mu0, occupancy=occupancy, delta=delta, sigma0=sigma0,
                                         conv_tol=conv_tol, opt_mu=opt_mu, mu_tol=mu_tol)

        self._finalize()
        logger.timer(self, 'DMFT', *cput0)

    def dmft(self, **kwargs):
        return self.kernel(**kwargs)

    def _kernel_mu(self, mu, occupancy, conv_tol):
        return kernel(self, mu, occupancy=occupancy, conv_tol=conv_tol)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)
        return self

    def run_diis(self, hyb, istep, adiis):
        if (adiis and istep >= self.diis_start_cycle):
            hyb = adiis.update(hyb)
            logger.debug1(self, 'DIIS for step %d', istep)
        return hyb

    def get_rdm_imp(self):
        '''Calculate the interacting local RDM from the impurity problem'''
        if self.solver_type == 'scf':
            return NotImplementedError
        elif self.solver_type == 'cc':
            return cc_rdm(self._scf, ao_orbs=range(self.nao))
        elif self.solver_type == 'fci':
            return fci_rdm(self._scf, ao_orbs=range(self.nao)) 

    def get_gf_imp(self, freqs, delta):
        '''Calculate the interacting local GF from the impurity problem'''
        if self.solver_type == 'scf':
            return mf_gf(self._scf, freqs, delta)
        elif self.solver_type == 'cc':
            return cc_gf(self._scf, freqs, delta, ao_orbs=range(self.nao),
                         gmres_tol=self.gmres_tol)
        elif self.solver_type == 'fci':
            #raise NotImplementedError
            assert (fci_)
            return fci_gf(self._scf, freqs, delta, ao_orbs=range(self.nao),
                          gmres_tol=self.gmres_tol)

    def get_gf0_imp(self, freqs, delta):
        '''Calculate the noninteracting local GF from the impurity problem'''
        himp = self._scf.get_hcore()
        nb = himp.shape[0]
        nw = len(freqs)
        sig_dum = np.zeros((nb,nb,nw,))
        gf = get_gf(himp, sig_dum, freqs, delta)
        return gf[:self.nao,:self.nao,:]

    def get_sigma_imp(self, freqs, delta):
        '''Calculate the local self-energy from the impurity problem'''
        gf0 = self.get_gf0_imp(freqs, delta)
        gf1 = self.get_gf_imp(freqs, delta)
        return get_sigma(gf0, gf1)


    def get_ldos_imp(self, freqs, delta):
        '''Calculate the local DOS from the impurity problem'''
        nw = len(freqs)
        nao = self.nao

        gf = self.get_gf_imp(freqs, delta)
        ldos = np.zeros([nao, nw])
        for p in range(nao):
           ldos[p,:] += -1./np.pi * np.imag(gf[p,p,:])
        return ldos

    def get_ldos_latt(self, freqs, delta, sigma=None):
        '''Calculate the local DOS from the lattice problem with a
        self-energy.'''
        nw = len(freqs)
        nao = self.nao
        nkpts = self.nkpts

        if sigma is None:
            sigma = self.get_sigma_imp(freqs, delta)
        gf = np.zeros([nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf += 1./nkpts * get_gf(self.hcore_k[k,:,:], sigma, freqs, delta)
        ldos = np.zeros([nao, nw])
        for p in range(nao):
           ldos[p,:] += -1./np.pi * np.imag(gf[p,p,:])
        return ldos

    # methods below have not been updated for present version of code

    def _gf (self, freqs, delta):
        assert (self.conv_)
        if self.solver_type != 'scf':
            assert (self.corr_ is not None)
        if self.solver_type == 'scf':
            gf = mf_gf (freqs, delta, \
                        self.mf_.mo_coeff, self.mf_.mo_energy)
        elif self.solver_type == 'cc':
            gf = cc_gf (freqs, delta, self.corr_, self.mf_.mo_coeff)
        elif self.solver_type == 'cc_ao':
            gf = cc_gf_ao (self.nao, freqs, delta, self.corr_, self.mf_.mo_coeff)
        elif self.solver_type == 'tdcc':
            gf = tdcc_gf (freqs, delta, self.corr_, self.mf_.mo_coeff)
        elif self.solver_type == 'tdcc_ao':
            gf = tdcc_gf_ao (self.nao, freqs, delta, self.corr_, self.mf_.mo_coeff)
        elif self.solver_type == 'fci':
            assert (fci_)
            gf = fci_gf (freqs, delta, self.corr_, self.mf_.mo_coeff)
        return gf[:self.nao,:self.nao,:]

    def get_lspectral (self, freqs, delta, sigma=None):
        nw = len(freqs)
        nao = self.nao
        nkpts = self.nkpts

        if sigma is None:
            sigma = self._local_sigma (freqs, delta)
        spec = np.zeros([nkpts, nao, nw])
        for k in range(nkpts):
            gf = get_gf(self.hcore_k[k,:,:], sigma, \
                        freqs, delta)
            for l in range(nao):
                spec[k,l,:] = -1./np.pi * np.imag(gf[l,l,:])
        return spec

    def get_lspectral_ni (self, freqs, delta):
        nw = len(freqs)
        nao = self.nao
        sigma = np.zeros([nao, nao, nw])
        return self.get_lspectral (freqs, delta, sigma)

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

