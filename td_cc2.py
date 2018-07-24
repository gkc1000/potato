import numpy as np
import scipy
import scipy.linalg
import scipy.integrate
import pyscf
import pyscf.ao2mo
import pyscf.gto as gto
import pyscf.scf as scf
import pyscf.cc.ccsd as ccsd
import pyscf.cc.eom_rccsd as eom_rccsd
import pyscf.ao2mo as ao2mo
import matplotlib.pyplot as plt

import greens_function
import single_particle

import tools
import time

# GKC fix inconsistent arg orders!!
def cc_gf_ao (nimp, freqs, delta, cc, mo_coeff):
    gf = greens_function.greens_function()
    # Calculate full (p,q) GF matrix in AO basis
    g_ip = gf.solve_ip_ao(cc, range(nimp), \
                          freqs.conj(), mo_coeff, delta).conj()
    g_ea = gf.solve_ea_ao(cc, range(nimp), \
                          freqs, mo_coeff, delta)

    return g_ip + g_ea

def cc_gf (freqs, delta, cc, mo_coeff):
    n = mo_coeff.shape[0]
    nw = len(freqs)
    gf = greens_function.greens_function()
    # Calculate full (p,q) GF matrix in MO basis
    g_ip = gf.solve_ip(cc, range(n), range(n), \
                       freqs.conj(), delta).conj()
    g_ea = gf.solve_ea(cc, range(n), range(n), \
                       freqs, delta)

    # Change basis from MO to AO
    gf = np.zeros([n, n, nw], np.complex128)
    for iw, w in enumerate(freqs):
        g_ip_ = np.dot(mo_coeff, np.dot(g_ip[:,:,iw], mo_coeff.T))
        g_ea_ = np.dot(mo_coeff, np.dot(g_ea[:,:,iw], mo_coeff.T))
        gf[:,:,iw] = g_ip_+g_ea_
    return gf


def get_ip(cc, norbs, times, tol):
    gf = greens_function.greens_function()
    # - sign determined empirically
    return -gf.td_ip(cc, range(norbs), range(norbs),
                     times, re_im="re", tol=tol)

def get_ea(cc, norbs, times, tol):
    gf = greens_function.greens_function()
    # - sign determined empirically
    return -gf.td_ea(cc, range(norbs), range(norbs),
                        times, re_im="re", tol=tol)

def get_ip_ao(cc, norbs, times, mo_coeff, tol):
    gf = greens_function.greens_function()
    # - sign determined empirically
    return -gf.td_ip_ao(cc, range(norbs), 
                        times, mo_coeff, re_im="re", tol=tol)

def get_ea_ao(cc, norbs, times, mo_coeff, tol):
    gf = greens_function.greens_function()
    # - sign determined empirically
    return -gf.td_ea_ao(cc, range(norbs), 
                        times, mo_coeff,
                        re_im="re", tol=tol)

def test(tf,nobs,U):
    nao = 2
    U = U
    htb = -1*tools.tb(nao)
    htb[0,0] = 0.01 # make it non-symmetric
    eri = np.zeros([nao,nao,nao,nao])
    eri[0,0,0,0] = U
    
    mol = gto.M()
    mol.build()
    mol.nelectron = nao #nao

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

    # 2*pi*max energy << tf-ti
    ti = 0; tf = tf; nobs = nobs
    times = np.linspace(ti,tf,nobs)
    deltat = float(tf - ti) / nobs

    # =================
    # single particle GF
    # ==================
    nocc = mol.nelectron / 2
    gip0 = single_particle.get_gip(mf.mo_energy, nocc, times)
    gea0 = single_particle.get_gea(mf.mo_energy, nocc, times)

    # =================
    # CC GF
    # ==================
    cc = ccsd.CCSD(mf)
    ecc = cc.ccsd()[0]
    print "CCSD corr = %20.12f" % (ecc)
    start = time.time()
    print "Solving lambda equations..."
    cc.solve_lambda()
    stop = time.time()
    print "Elapsed time for CC", stop-start

    # This tolerance controls the accuracy of the
    # RK45 integrator used in the green's function algorithm
    # Each element will be computed to an accuracy of tol
    tol=1.e-5

    start = time.time()
    gip = get_ip_ao(cc, 1, times, mf.mo_coeff, tol)
    gea = get_ea_ao(cc, 1, times, mf.mo_coeff, tol)
    stop = time.time()
    print "Elasped time for TD-propagation", stop-start
    
    # This is the difference between the exact MF and CC
    # Green's functions. If you dial tol up, this will go to 0
    print "IP difference", np.linalg.norm(gip-gip0)
    print "EA difference", np.linalg.norm(gea-gea0)

    # predict out to long times
    # note ntotal must be 2**n+1 since
    # we use romberg integration to do fourier transform integral
    tmax0 = 10000
    ntotal0 = tmax0 / deltat

    nbase2 = np.int(np.log(ntotal0)/np.log(2))
    ntotal = 2**nbase2+1

    # 2*pi/tmax gives a minimum oscillation frequency, so
    # graph will wiggle at least on this scale
    print "Total propagation time: ", ntotal * deltat

    print "Predict ip0 ============"

    predicted_gf_ip0 = tools.predict_gf(gip0, ntotal)
    predicted_gf_ea0 = tools.predict_gf(gea0, ntotal)

    print "Predict ip ============"
    start = time.time()

    predicted_gf_ip = tools.predict_gf(gip, ntotal)
    predicted_gf_ea = tools.predict_gf(gea, ntotal)

    stop = time.time()
    print "Elasped time for prediction", stop-start

    print "Norm difference", np.linalg.norm(gip-gip0)
    #ddd

    # transform time-domain to frequency domain
    gret0 = -1j * (predicted_gf_ip0 + predicted_gf_ea0)
    gret_ao0 = np.einsum("pi,ijt,jq->pqt", mf.mo_coeff, gret0,
                        mf.mo_coeff.T)

    # old stuff
    # gret = -1j * (predicted_gf_ip + predicted_gf_ea)
    # gret_ao = np.einsum("pi,ijt,jq->pqt", mf.mo_coeff, gret,
    #                     mf.mo_coeff.T)

    gret_ao = -1j * (predicted_gf_ip + predicted_gf_ea)

    extrapolated_times = np.array([deltat*i for i in range(ntotal)])
    tmax = extrapolated_times[-1]

    # for i in range(gret_ao.shape[2]):
    #     print i, extrapolated_times[i], gret_ao0[0,0,i], gret_ao[0,0,i]
    print np.linalg.norm(gret_ao0-gret_ao)

    # compute freq. dependent GF
    freqs = np.linspace(-10, +10, 400)
    delta = 1.e-1 # this should be on the scale of pi/tmax

    # tdgf_w0 = tools.get_gfw(gret_ao0, extrapolated_times,
    #                         freqs, delta)
    start = time.time()
    tdgf_w = tools.get_gfw(gret_ao, extrapolated_times,
                            freqs, delta)
    stop = time.time()
    print "Elapsed time: FT", stop-start

    start = time.time()
    # w_gf_w = cc_gf(freqs, delta, cc, mf.mo_coeff)
    # "1" is the number of impurities
    w_gf_w = cc_gf_ao(1, freqs, delta, cc, mf.mo_coeff)
    stop = time.time()
    print "Elapsed time: frequency CC", stop-start
    # when you see the peak, in addition to the broadening
    # there is also a small real shift (on the size of the broadening);
    # I don't fully get why it's there (maybe it's the use of the Gaussian
    # form of broadening?)
    #plt.plot(freqs, -1./np.pi*tdgf_w0[0,0].imag, "rx-")
    plt.plot(freqs, -1./np.pi*tdgf_w[0,0].imag, "b+-")

    plt.plot(freqs, -1./np.pi*w_gf_w[0,0].imag, "g*-")
    plt.savefig("AO_"+str(U)+"_"+str(tf)+"_"+str(nobs)+".pdf")
    plt.close()

    #plt.show()



def test_all():
    test(40,800,8)
    # for U in [8]:
    #     for tf in [80]:
    #         for nobs in [800]:
    #             print U, tf, nobs
    #             test(tf, nobs, U)
                
