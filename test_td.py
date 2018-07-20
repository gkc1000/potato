#!/usr/bin/python

import math
import numpy as np
import scipy
import scipy.linalg
inv = scipy.linalg.inv

import pyscf
import pyscf.gto as gto
import pyscf.scf as scf
import pyscf.cc.ccsd as ccsd
import pyscf.cc.eom_rccsd as eom_rccsd
import pyscf.ao2mo as ao2mo

import greens_function
import numint_

import matplotlib.pyplot as plt

def _get_linear_freqs(wl, wh, nw):
    freqs = np.linspace(wl, wh, nw) 
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)
    return freqs, wts


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


def cc_td_gf(ti, tf, times, cc_eom, mo_coeff):
    n = mo_coeff.shape[0]
    nt = len(times)

    gip = np.zeros((n,n,nt), np.complex128)
    gea = np.zeros((n,n,nt), np.complex128)
    gf = greens_function.greens_function()

    # Calculate full (p,q) GF matrix in MO basis
    g_ip = gf.td_ip(cc_eom, range(n), range(n), \
                    ti, tf, times)

    g_ea = gf.td_ea(cc_eom, range(n), range(n), \
                    ti, tf, times)
    
    # Change basis from MO to AO
    # Compute retarded GF
    # Defn. Eqn. A.5b, pg. 141 https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf 
    gf_ret_ao = np.zeros([n, n, nt], np.complex128)

    for i in range(nt):
        g_ip_ao = np.dot(mo_coeff, np.dot(g_ip[:,:,i], mo_coeff.T))
        g_ea_ao = np.dot(mo_coeff, np.dot(g_ea[:,:,i], mo_coeff.T))
        gf_ret_ao[:,:,i] = -1j*(g_ip_ao+g_ea_ao) # note theta fn is unnecessary if evolve for +ve time

    return gf_ret_ao


def test_td():
    nao = 2
    U = 0.

    htb = -1*_tb(nao)
    htb[0,0]=0.0
    eri = np.zeros([nao,nao,nao,nao])
    for k in range(nao):
        eri[k,k,k,k] = U

    delta = 0.01
    
    mol = gto.M()
    mol.build()
    mol.nelectron = 2 #nao

    mf = scf.RHF(mol)
    mf.verbose = 0
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

    ti=0
    tf=400

    nquad=12
    times=np.linspace(ti, tf, 2**nquad+1)
    gf_ret = cc_td_gf (ti, tf, times, cc, mf.mo_coeff)

    freqs_ = _get_linear_freqs(-6, 6, 512)[0]

    gf_ret_w = np.zeros([gf_ret.shape[0],gf_ret.shape[1],len(freqs_)],dtype=np.complex128)

    halftime=(2**(nquad-1)+1)
    inttimes = times[:halftime]
    delta=0.1
    for iw, w in enumerate(freqs_):
        ftwts = 1./(tf-ti)*np.array([np.exp(1j*(w*t))*np.exp(-delta**2*t) for t in times[:halftime]], dtype=np.complex128)
        for p in range(gf_ret.shape[0]):
            for q in range(gf_ret.shape[1]):
                gfft = gf_ret[p,q,:halftime] * ftwts
                gf_ret_w[p,q,iw]=scipy.integrate.romb(gfft)

    print gf_ret_w

    dos = np.zeros([freqs_.shape[0]])
    for k in range(nao):
        dos[:] += 1./np.pi * np.imag(gf_ret_w[k,k,:])

    plt.plot(freqs_, dos)
    plt.show()

