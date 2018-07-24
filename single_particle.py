import numpy as np
import scipy
import scipy.linalg
import scipy.integrate
import pyscf
import pyscf.ao2mo
import pyscf.gto as gto
import pyscf.scf as scf
import pyscf.lib.diis
import matplotlib.pyplot as plt

import tools

def get_gip(eigs, nocc, times):
    n = eigs.shape[0]
    nvirt = n - nocc

    nt = len(times)
    gip = np.zeros([n,n,nt], np.complex128)

    for i in range(nocc):
        gip[i,i,:] = np.exp(-1j*eigs[i]*times)
    return gip

def get_gea(eigs, nocc, times):
    n = eigs.shape[0]
    nvirt = n - nocc

    nt = len(times)
    gea = np.zeros([n,n,nt], np.complex128)

    for a in range(nocc, n):
        gea[a,a,:] = np.exp(-1j*eigs[a]*times)
    return gea


def test():
    nao = 2

    htb = -1*tools.tb(nao)
    htb[0,0]=0.0
    eri = np.zeros([nao,nao,nao,nao])
    
    mol = gto.M()
    mol.build()
    mol.nelectron = 2 #nao

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_memory = 1000
    mf.get_hcore = lambda *args: htb
    mf.get_ovlp = lambda *args: np.eye(nao)
    mf._eri = pyscf.ao2mo.restore(8, eri, nao)
    mf.init_guess = '1e'
    mf.scf()

    print 'MF energy = %20.12f' % (mf.e_tot)
    print 'MO energies :'
    print mf.mo_energy
    print '----\n'

    # first do a propagation out for a small amount of time.
    # We will use this to seed the linear prediction later
    ti = 0; tf = 10; nobs = 100
    times = np.linspace(ti,tf,nobs)
    deltat = float(tf - ti) / nobs
    
    nocc = mol.nelectron / 2
    gip = get_gip(mf.mo_energy, nocc, times)
    gea = get_gea(mf.mo_energy, nocc, times)

    # predict out to long times
    # note ntotal must be 2**n+1 since
    # we use romberg integration to do fourier transform integral
    tmax0 = 1000
    ntotal0 = tmax0 / deltat

    nbase2 = np.int(np.log(ntotal0)/np.log(2))
    ntotal = 2**nbase2+1
    
    # 2*pi/tmax gives a minimum oscillation frequency, so
    # graph will wiggle at least on this scale
    print "Total propagation time: ", ntotal * deltat
    
    predicted_gf_ip = tools.predict_gf(gip, ntotal)
    predicted_gf_ea = tools.predict_gf(gea, ntotal)
    
    gret = -1j * (predicted_gf_ip + predicted_gf_ea)
    gret_ao = np.einsum("pi,ijt,jq->pqt", mf.mo_coeff, gret, mf.mo_coeff.T)
    
    extrapolated_times = np.array([deltat*i for i in range(ntotal)])
    tmax = extrapolated_times[-1]

    # plot the ip real and imaginary parts. For this simple problem
    # this is just a single exponential.
    plt.plot(extrapolated_times, predicted_gf_ip[0,0].real)
    plt.plot(extrapolated_times, predicted_gf_ip[0,0].imag)
    plt.show()

    # compute freq. dependent GF
    freqs = np.linspace(-1.2, -0.8, 1000)
    delta = 3.e-3 # this should be on the scale of pi/tmax

    gf_w = tools.get_gfw(gret_ao, extrapolated_times,
                         freqs, delta)

    # when you see the peak, in addition to the broadening
    # there is also a small real shift (on the size of the broadening);
    # I don't fully get why it's there (maybe it's the use of the Gaussian
    # form of broadening?)
    plt.plot(freqs, -1./np.pi*gf_w[0,0].imag)
    plt.show()

    
