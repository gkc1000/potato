import math
import numpy as np
import scipy
import scipy.linalg
inv = scipy.linalg.inv

import pyscf
import pyscf.gto as gto
import pyscf.scf as scf
import pyscf.cc.ccsd as ccsd
import pyscf.cc.rccsd_eom as rccsd_eom
import pyscf.ao2mo as ao2mo

import greens_function
import numint_

import matplotlib.pyplot as plt

from sys import path
path.append('/home/carlosjh/other/CheMPS2/PyCheMPS2/build/lib.linux-x86_64-2.7/')
import PyCheMPS2
import ctypes


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

def mf_gf (freqs, delta, mo_coeff, mo_energy, nocc):
    nw = len(freqs)
    n = mo_coeff.shape[0]
    gf = np.zeros([n, n, nw], np.complex128)
    for iw, w in enumerate(freqs):
        g_ip = np.diag(1./((w+1j*delta) * \
                np.ones([nocc],np.complex128) - mo_energy[:nocc]))
        g_ea = np.diag(1./((w+1j*delta) * \
                np.ones([n-nocc],np.complex128) - mo_energy[nocc:]))
        g_ip_ = np.dot(mo_coeff[:,:nocc], np.dot(g_ip, \
                                                 mo_coeff[:,:nocc].T))
        g_ea_ = np.dot(mo_coeff[:,nocc:], np.dot(g_ea, \
                                                 mo_coeff[:,nocc:].T))
        gf[:,:,iw] = g_ip_+g_ea_
    return gf

def cc_gf (freqs, delta, cc_eom, mo_coeff, mo_energy):
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

def fci_gf (freqs, delta, mo_coeff, energy_gs, gs_vector, \
            HamCheMPS2, theFCI):
    n  = mo_coeff.shape[0]
    nw = len(freqs)
    gf = np.zeros([n, n, nw], np.complex128)

    orbsLeft  = np.arange(n, dtype=ctypes.c_int)
    orbsRight = np.arange(n, dtype=ctypes.c_int)

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

def fci_sol (h0, h1, eri, nel):
    norb = h1.shape[0]
    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    Group = 0
    orbirreps = np.zeros((norb,), dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(norb, Group, orbirreps)
    HamCheMPS2.setEconst( h0 )
    for cnt1 in range(norb):
        for cnt2 in range(norb):
            HamCheMPS2.setTmat(cnt1, cnt2, h1[cnt1,cnt2])
            for cnt3 in range(norb):
                for cnt4 in range(norb):
                    HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, eri[cnt1,cnt3,cnt2,cnt4])

    assert( nel % 2 == 0 )
    Nel_up       = nel / 2
    Nel_down     = nel / 2
    Irrep        = 0
    maxMemWorkMB = 100.0
    FCIverbose   = 0
    theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose )
    GSvector = np.zeros( [ theFCI.getVecLength() ], dtype=ctypes.c_double )
    GSvector[ theFCI.LowestEnergyDeterminant() ] = 1 # Large component for quantum chemistry
    EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
    return HamCheMPS2, theFCI, GSvector, EnergyCheMPS2

def test():
    nao = 10
    U = 1.0

    solver = 'fci'  # 'scf', 'cc', 'fci'

    htb = -1*_tb(nao)
    eri = np.zeros([nao,nao,nao,nao])
    for k in range(nao):
        eri[k,k,k,k] = U
    delta = _get_delta(htb)

    mol = gto.M()
    mol.build()
    mol.nelectron = 2 #nao

    mf = scf.RHF(mol)
    mf.verbose = 0
    # mf.verbose = 4
    mf.max_memory = 1000

    mf.get_hcore = lambda *args: htb
    mf.get_ovlp = lambda *args: np.eye(nao)
    mf._eri = ao2mo.restore(8, eri, nao)
    mf.init_guess = '1e'
    mf.scf()

    print 'MF energy = %20.12f' % (mf.e_tot)
    print 'MO energies :'
    print mf.mo_energy
    print '----\n'

    HamCheMPS2, theFCI = None, None
    if solver == 'cc':
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
        print '====\n'

        #cc_eom.t1 = cc.t1
        #cc_eom.t2 = cc.t2
        cc_eom.l1 = cc.l1
        cc_eom.l2 = cc.l2

        print 'CC IP evals'
        print cc_eom.ipccsd()[0]
        print 'CC EA evals'
        print cc_eom.eaccsd()[0]

    elif solver == 'fci':
        h0   = 0.
        h1t  = np.dot(mf.mo_coeff.T, np.dot(htb, mf.mo_coeff))
        erit = ao2mo.incore.full(mf._eri, mf.mo_coeff, compact=False)
        erit = erit.reshape([nao,nao,nao,nao])

        HamCheMPS2, theFCI, GSvector, en_FCIgs = \
                fci_sol (h0, h1t, erit, mol.nelectron)
        print "FCI corr = %20.12f" % (en_FCIgs-mf.e_tot)

    evals, evecs = scipy.linalg.eigh(htb)

    mu = ( mf.mo_energy[mol.nelectron//2-1] + \
           mf.mo_energy[mol.nelectron//2] )/2.
    #mu += 0.05
    nocc = mol.nelectron//2

    delta_ = 1.e-4
    def _gf (w, delta):
        if solver == 'scf':
            return mf_gf (w, delta, mf.mo_coeff, mf.mo_energy, nocc)
        elif solver == 'cc':
            return cc_gf (w, delta, cc_eom, mf.mo_coeff, mf.mo_energy)
        elif solver == 'fci':
            return fci_gf (w, delta, mf.mo_coeff, en_FCIgs, GSvector, \
                           HamCheMPS2, theFCI)
    def _mf_gf (w, delta):
        return mf_gf (w, delta, evecs, evals, nocc)

    freqs_ = _get_linear_freqs(-10., 10., 64)[0]
    gf0 = mf_gf (freqs_, delta, evecs, evals, nocc)
    gf1 = _gf (freqs_, delta)
    dos0 = np.zeros([freqs_.shape[0]])
    dos1 = np.zeros([freqs_.shape[0]])
    for k in range(nao):
       dos0[:] += -1./np.pi * np.imag(gf0[k,k,:])
       dos1[:] += -1./np.pi * np.imag(gf1[k,k,:])

    plt.plot(freqs_, dos0)
    plt.plot(freqs_, dos1)
    plt.show()

    def _eval_p(w, delta):
        gf_ = _gf(np.array([w]), delta)
        return gf_[:,:,0]
    def _eval_n(w, delta):
        return np.trace(_eval_p(w, delta))

    mf_infi = _mf_gf(np.array([1j*1000000.+mu]), delta_)
    gf_infi = _gf(np.array([1j*1000000.+mu]), delta_)
    sigma_infi = get_sigma(mf_infi, gf_infi)[:,:,0]

    mf_infr = _mf_gf(np.array([1000000.]), delta_)
    gf_infr = _gf(np.array([1000000.]), delta_)
    sigma_infr = get_sigma(mf_infr, gf_infr)[:,:,0]

    def _eval_en0(w, delta):
        gf_ = _gf(np.array([w]), delta)
        return np.trace(np.dot(htb, gf_[:,:,0]))
    def _eval_en1(w, delta):
        gf_ = _gf(np.array([w]), delta)
        if np.iscomplex(w):
            return np.trace(np.dot(sigma_infi, gf_[:,:,0]))
        else:
            return np.trace(np.dot(sigma_infr, gf_[:,:,0]))
    def _eval_en2(w, delta):
        mf_ = _mf_gf(np.array([w]), delta)
        gf_ = _gf(np.array([w]), delta)
        sigma = get_sigma(mf_, gf_)
        if np.iscomplex(w):
            return np.trace(np.dot(sigma[:,:,0]-sigma_infi, gf_[:,:,0]))
        else:
            return np.trace(np.dot(sigma[:,:,0]-sigma_infr, gf_[:,:,0]))

    lplt = False

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

    lr = False

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

