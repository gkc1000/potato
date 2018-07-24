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

from tools import *

def get_ip(cc, norbs, times, tol):
    gf = greens_function.greens_function()
    # - sign determined empirically
    return -gf.td_ip(cc, range(norbs), range(norbs),
                     times, tol)

def get_ea(cc, norbs, times, tol):
    gf = greens_function.greens_function()
    # - sign determined empirically
    return -gf.td_ea(cc, range(norbs), range(norbs),
                     times, tol)

def test():
    nao = 2

    htb = -1*tb(nao)
    htb[0,0] = 0.01 # make it non-symmetric
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
    
    ti = 0; tf = 10; nobs = 8
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
    print "Solving lambda equations..."
    cc.solve_lambda()

    # This tolerance controls the accuracy of the
    # RK45 integrator used in the green's function algorithm
    # Each element will be computed to an accuracy of tol
    tol=1.e-5
    gip = get_ip(cc, nao, times, tol)
    gea = get_ea(cc, nao, times, tol)

    # This is the difference between the exact MF and CC
    # Green's functions. If you dial tol up, this will go to 0
    print "IP difference", np.linalg.norm(gip-gip0)
    print "EA difference", np.linalg.norm(gea-gea0)
    
