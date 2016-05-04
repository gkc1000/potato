#!/usr/bin/env python

import sys
import scipy
import numpy
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
import pyscf.cc.rccsd_eom
import greens_function
import pyscf.cc

CISD = False

def read_hubbard_params(nbath,ncorr,filename):
    basename = filename.split("/")[-1]
    U = float(basename.split("_")[0].split("u")[1])
    t = 1.0

    fin = open(filename, "r")
    count = 0
    index = 0
    bath_v = numpy.empty((ncorr,nbath),dtype=float)
    for line in fin.readlines():
        if count == 3: # we are at the orbital energies
            bath_orbitals = [float(x) for x in line.strip(" \t").split()]
            print bath_orbitals
        if count >= 5 and index < ncorr:
            bath_v[index] = [float(x) for x in line.strip(" \t").split()]
            print bath_v[index]
            index += 1
        count += 1
    return U, t, bath_orbitals, bath_v

def main():
    args = sys.argv[1:]
    if len(args) == 2:
        U = int(args[0])
        eta = float(args[1])
    else:
        print "Usage: U/t eta[au]"
        return -1

    nbath = 8
    ncorr = 2
    nelectron = ncorr + nbath
    nbas = ncorr + nbath

    if U == 1:
        filename = "params/u1.00_6th.mak"
    elif U == 2:
        filename = "params/u2.00_10th.mak"
    elif U == 3:
        filename = "params/u3.00_13th.mak"
    elif U == 4:
        filename = "params/u4.00_12th.mak"
    elif U == 5:
        filename = "params/u5.00_11th.mak"
    elif U == 7:
        filename = "params/u7.00_10th.mak"
    elif U == 9:
        filename = "params/u9.00_12th.mak"
    else:
        print "U not coded -- quitting."
        return -1

    U, t, bath_onsite, bath_v = read_hubbard_params(nbath, ncorr, filename)

    h1 = np.zeros((nbas,nbas))
    # Filling in the correlated orbital matrix elements
    h1[0,0] = h1[1,1] = -U/2.
    h1[0,1] = h1[1,0] = -t

    # Filling in the bath orbital matrix elements
    for i in range(nbath):
        index = ncorr + i
        h1[index,index] = bath_onsite[i]
        for j in range(ncorr):
            h1[j,index] = bath_v[j,i]
            h1[index,j] = h1[j,index]

    numpy.set_printoptions(threshold=100,precision=6,linewidth=120)

    for i in range(-1,nbas):
        for j in range(nbas):
            if i == -1:
                if j < ncorr:
                    print "%7s" % "corr",
                else:
                    print "%7s" % "bath",
            else:
                print "%7.4f" % h1[i,j],
        print ""
    print ""

    eri = numpy.zeros((nbas,nbas,nbas,nbas))
    for i in range(ncorr):
        eri[i,i,i,i] = U

    # Interfacing with pyscf
    # ----------------------
    mol = gto.M()
    mol.nelectron = nelectron
    mol.incore_anyway = True

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(nbas)
    mf._eri = ao2mo.restore(8, eri, nbas)

    mf.init_guess = '1e'
    escf = mf.scf()
    print "# pyscf HF evals = ", mf.mo_energy
    print "# pyscf HF evecs = "
    print mf.mo_coeff

    cc = pyscf.cc.CCSD(mf)
    print "cc.mo_energy =", cc.mo_energy
    ecc,t1,t2 = cc.ccsd()
    print "CCSD corr   : %.15f" % ecc
    print "CCSD energy : %.15f" % (ecc+escf)

    print "Solving lambda equations..."
    conv,l1,l2 = cc.solve_lambda()

    # Interfacing with EOM pyscf
    # ----------------------
    print "Repeating with EOM CCSD"
    cc_eom = pyscf.cc.rccsd_eom.RCCSD(mf) 
    def my_ao2mofn(mol, bas, compact):
        return change_basis_2el(eri, mf.mo_coeff)
    # Give _ERIS class the eris in the MO basis
    eris = pyscf.cc.rccsd_eom._ERIS(cc_eom, ao2mofn=my_ao2mofn)
    ecc_eom, t1_eom, t2_eom = cc_eom.ccsd(eris=eris)
    print "EOM-CCSD corr   : %.15f" % ecc_eom
    print "EOM-CCSD energy : %.15f" % (ecc_eom+escf)

    #cc_eom.t1 = cc.t1
    #cc_eom.t2 = cc.t2
    cc_eom.l1 = cc.l1
    cc_eom.l2 = cc.l2

    if CISD == True:
        cc_eom.t1 *= 1e-5
        cc_eom.t2 *= 1e-5
        cc_eom.l1 *= 1e-5
        cc_eom.l2 *= 1e-5

    dw = 0.03
    wmin = -8.0
    wmax = 8.0
    nw = int((wmax-wmin)/dw) + 1
    omegas = numpy.linspace(wmin, wmax, nw)
    gip = np.zeros((nbas,nbas,len(omegas)),np.complex)
    gea = np.zeros((nbas,nbas,len(omegas)),np.complex)
    gf = greens_function.greens_function()
    # Calculate full (p,q) GF matrix in MO basis
    gip, gea = gf.solve_gf(cc_eom,range(nbas),range(nbas),omegas,eta)

    # Change basis from MO to AO
    gip_ao = np.einsum('ip,pqw,qj->ijw',mf.mo_coeff,gip,mf.mo_coeff.T)
    gea_ao = np.einsum('ip,pqw,qj->ijw',mf.mo_coeff,gea,mf.mo_coeff.T)

    # Save the local GF for the "correlated" impurity orbitals
    for i in range(ncorr):
        numpy.savetxt("gf_U-%.1f_ao-%d%d.dat"%(U,i,i), 
            numpy.column_stack([omegas,
                                gip_ao[i,i,:].real, gip_ao[i,i,:].imag,
                                gea_ao[i,i,:].real, gea_ao[i,i,:].imag]))

    # Make G0 for sigma
    nocc = nelectron/2
    e,c = eig(h1)
    #e,c = mf.mo_energy.copy(), mf.mo_coeff.copy()
    g0ip = np.zeros_like(gip)
    g0ea = np.zeros_like(gea)
    for iw,w in enumerate(omegas):
        for i in range(nocc):
            g0ip[i,i,iw] = 1./(w-e[i]-1j*eta)
        for a in range(nocc,nbas):
            g0ea[a,a,iw] = 1./(w-e[a]+1j*eta)
    # Change basis from MO to AO
    g0ip_ao = np.einsum('ip,pqw,qj->ijw',c,g0ip,c.T)
    g0ea_ao = np.einsum('ip,pqw,qj->ijw',c,g0ea,c.T)

    for i in range(1):
    #for i in range(ncorr):
        numpy.savetxt("gf0_U-%.1f_ao-%d%d.dat"%(U,i,i), 
            numpy.column_stack([omegas,
                                g0ip_ao[i,i,:].real, g0ip_ao[i,i,:].imag,
                                g0ea_ao[i,i,:].real, g0ea_ao[i,i,:].imag]))

    g0_ret_ao = g0ip_ao.conj() + g0ea_ao
    g_ret_ao = gip_ao.conj() + gea_ao
    sigma = np.zeros_like(g_ret_ao)
    for iw,w in enumerate(omegas):
        sigma[:,:,iw] = np.linalg.inv(g0_ret_ao[:,:,iw]) - np.linalg.inv(g_ret_ao[:,:,iw]) 

    for i in range(1):
    #for i in range(ncorr):
        numpy.savetxt("sigma_U-%.1f_ao-%d%d.dat"%(U,i,i), 
            numpy.column_stack([omegas, sigma[i,i,:].real, sigma[i,i,:].imag]))

def eig(H,S=None):
    """
    Diagonalize a real, symmetrix matrix and return sorted results.
    
    Return the eigenvalues and eigenvectors (column matrix) 
    sorted from lowest to highest eigenvalue.
    """
    E,C = scipy.linalg.eigh(H,S)
    E = np.real(E)
    C = np.real(C)

    idx = E.argsort()
    E = E[idx]
    C = C[:,idx]

    return E,C

def change_basis_2el(g,C):
    """Change basis for 2-el integrals and return.

    - C is a matrix (Ns x Nnew) whose columns are new basis vectors,
      expressed in the basis in which g is given.
    - Typical operation is g in the site basis, and C is a 
      transformation from site to some-other-basis.
    """
    g1 = np.tensordot(C,g,axes=[0,3])
    g1 = np.transpose(g1,(1,2,3,0))
    # g1 is Ns x Ns x Ns x Nnew
    g = np.tensordot(C,g1,axes=[0,2])
    g = np.transpose(g,(1,2,0,3))
    # g is Ns x Ns x Nnew x Nnew
    g1 = np.tensordot(C,g,axes=[0,1])
    g1 = np.transpose(g1,(1,0,2,3))
    # g1 is Ns x Nnew x Nnew x Nnew
    g = np.tensordot(C,g1,axes=[0,0])
    # g is Nnew x Nnew x Nnew x Nnew
    return g

if __name__ == '__main__':
    main()

