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

def get_sigma(freqs, mf_gf, corr_gf):
    nw = len(freqs)
    sigma = np.zeros_like(mf_gf)
    for iw in range(nw):
        sigma[:,:,iw] = inv(mf_gf[:,:,iw]) - inv(corr_gf[:,:,iw])
    return sigma

def mf_gf (freqs, delta, mo_coeff, mo_energy):
    nw = len(freqs)
    n = mo_coeff.shape[0]
    gf = np.zeros([n, n, nw], np.complex128)
    for iw, w in enumerate(freqs):
        resolvent = np.diag(1./((w+1j*delta) * \
                            np.ones([n],np.complex128) - mo_energy))
        gf[:,:,iw] = np.dot(mo_coeff, np.dot(resolvent, \
                                             mo_coeff.T))
    return gf

def cc_gf (freqs, delta, cc_eom, mo_coeff, mo_energy):
    n = mo_coeff.shape[0]
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
        gip_ao[:,:,iw] = np.dot(mo_coeff, np.dot(gip[:,:,iw], \
                                                 mo_coeff.T))
        gea_ao[:,:,iw] = np.dot(mo_coeff, np.dot(gea[:,:,iw], \
                                                 mo_coeff.T))
    return gip_ao.conj()+gea_ao

def test():
    nao = 10
    U = 1.6

    htb = -1*_tb(nao)
    eri = np.zeros([nao,nao,nao,nao])
    for k in range(nao):
        eri[k,k,k,k] = U
    delta = _get_delta(htb)

    mol = gto.M()
    mol.build()
    mol.nelectron = nao

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

    evals, evecs = scipy.linalg.eigh(htb)

    print 'CC IP evals'
    print cc_eom.ipccsd()[0]
    print 'CC EA evals'
    print cc_eom.eaccsd()[0]

    delta_ = 1.e-4
    def _gf (w, delta):
        # return mf_gf (w, delta, mf.mo_coeff, mf.mo_energy)
        return cc_gf (w, delta, cc_eom, mf.mo_coeff, mf.mo_energy)
    def _sigma (w, delta):
        gf0 = mf_gf (w, delta, evecs, evals)
        gf1 = _gf (w, delta)
        return get_sigma (w, gf0, gf1)

    # freqs_, wts_ = _get_linear_freqs(-10., 10., 64)
    # gf0 = mf_gf (freqs_, delta, evecs, evals)
    # gf1 = _gf (freqs_, delta)
    # dos0 = np.zeros([freqs_.shape[0]])
    # dos1 = np.zeros([freqs_.shape[0]])
    # for k in range(nao):
    #     dos0[:] += -1./np.pi * np.imag(gf0[k,k,:])
    #     dos1[:] += -1./np.pi * np.imag(gf1[k,k,:])

    # plt.plot(freqs_, dos0)
    # plt.plot(freqs_, dos1)
    # plt.show()

    def _eval_p(w, delta):
        return _gf([w], delta)
    def _eval_n(w, delta):
        return np.trace(_eval_p(w, delta)[:,:,0])

    sigma_inf = _sigma([1000000.], delta)
    def _eval_en0(w, delta):
        return np.trace(np.dot(sigma_inf[:,:,0], \
                               _gf([w], delta)[:,:,0]))
    def _eval_en1(w, delta):
        return np.trace(np.dot(htb, _gf([w], delta)[:,:,0]))
    def _eval_enc(w, delta):
        sigma = _sigma([w], delta)
        sigma -= sigma_inf
        return np.trace(np.dot(sigma[:,:,0], _gf([w], delta)[:,:,0]))

    mu = ( mf.mo_energy[mol.nelectron//2-1] + \
           mf.mo_energy[mol.nelectron//2] )/2.

    # NL = # poles to left of mu, NR = # poles to right of mu
    # nao = NL + NR
    # integration gives NR - NL (factor of 2 in imag_fn)
    print 'number of electrons'
    nint_n = numint_.int_quad_real (_eval_n, mu, x0=-40., \
                                    epsrel=1.0e-6, delta=delta_)
    print 'nint_n [real] = ', 2*nint_n
    nint_n = numint_.int_quad_imag (_eval_n, mu, \
                                    epsrel=1.0e-6, delta=delta_)
    nint_n = 2*0.5*(nao-nint_n)
    print 'nint_n [imag] = ', nint_n
    # additional factor of 2 by spin integration
    print '----\n'

    print 'energy'
    # energy due to a constant self-energy
    nint_e0 = numint_.int_quad_real (_eval_en0, mu, x0=-40., \
                                     epsrel=1.0e-6, delta=delta_)
    print 'nint e0 [real] = ', nint_e0
    nint_e0 = numint_.int_quad_imag (_eval_en0, mu, \
                                     epsrel=1.0e-6, delta=delta_)
    e0 = (np.real(np.trace(sigma_inf[:,:,0])) - nint_e0)
    print 'nint e0 [imag] = ', e0/2

    # trace of h with GF
    nint_e1 = numint_.int_quad_real (_eval_en1, mu, x0=-40., \
                                     epsrel=1.0e-6, delta=delta_)
    print 'nint e1 [real] = ', 2*nint_e1
    nint_e1 = numint_.int_quad_imag (_eval_en1, mu, \
                                     epsrel=1.0e-6, delta=delta_)
    print 'nint e1 [imag] = ', -nint_e1

    # energy due to 1/w self-energy
    nint_ec = numint_.int_quad_real (_eval_enc, mu, x0=-40., \
                                     epsrel=1.0e-6, delta=delta_)
    print 'nint ec [real] = ', nint_ec
    nint_ec = numint_.int_quad_imag (_eval_enc, mu, \
                                     epsrel=1.0e-6, delta=delta_)
    print 'nint ec [imag] = ', -nint_ec/2.
    print 'nint_e = ', -nint_e1 + e0/2. -nint_ec/2.
    print '----\n'

