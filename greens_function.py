import collections

import numpy as np
import gminres

import pyscf
import pyscf.cc
from pyscf.cc.eom_rccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea
###################
# EA Greens       #
###################

def greens_b_vector_ea_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to minus
        vector1 += -cc.t1[p,:]
        vector2 += -cc.t2[p,:,:,:]
    else:
        vector1[ p-nocc ] = 1.0
    return amplitudes_to_vector_ea(vector1,vector2)

def greens_e_vector_ea_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to plus
        vector1 += cc.l1[p,:]
        vector2 += (2*cc.l2[p,:,:,:] - cc.l2[:,p,:,:])
        pass
    else:
        vector1[ p-nocc ] = -1.0
        vector1 += np.einsum('ia,i->a', cc.l1, cc.t1[:,p-nocc])
        vector1 += 2*np.einsum('klca,klc->a', cc.l2, cc.t2[:,:,:,p-nocc])
        vector1 -=   np.einsum('klca,lkc->a', cc.l2, cc.t2[:,:,:,p-nocc])

        vector2[:,p-nocc,:] += -2.*cc.l1
        vector2[:,:,p-nocc] += cc.l1
        vector2 += 2*np.einsum('k,jkba->jab', cc.t1[:,p-nocc], cc.l2)
        vector2 -=   np.einsum('k,jkab->jab', cc.t1[:,p-nocc], cc.l2)
    return amplitudes_to_vector_ea(vector1,vector2)

###################
# IP Greens       #
###################

def greens_b_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p < nocc:
        vector1[ p ] = 1.0
    else:
        vector1 += cc.t1[:,p-nocc]
        vector2 += cc.t2[:,:,:,p-nocc]
    return amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p < nocc:
        vector1[ p ] = -1.0
        vector1 += np.einsum('ia,a->i', cc.l1, cc.t1[p,:])
        vector1 += 2*np.einsum('ilcd,lcd->i', cc.l2, cc.t2[p,:,:,:])
        vector1 -=   np.einsum('ilcd,ldc->i', cc.l2, cc.t2[p,:,:,:])

        vector2[p,:,:] += -2.*cc.l1
        vector2[:,p,:] += cc.l1
        vector2 += 2*np.einsum('c,ijcb->ijb', cc.t1[p,:], cc.l2)
        vector2 -=   np.einsum('c,jicb->ijb', cc.t1[p,:], cc.l2)
    else:
        vector1 += -cc.l1[:,p-nocc]
        vector2 += -2*cc.l2[:,:,p-nocc,:] + cc.l2[:,:,:,p-nocc]

    return amplitudes_to_vector_ip(vector1,vector2)

def greens_func_multiply(ham,vector,linear_part,args=None):
    return np.array(ham(vector) + (linear_part)*vector)

def initial_ip_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    return amplitudes_to_vector_ip(vector1,vector2)

def initial_ea_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=complex)
    return amplitudes_to_vector_ea(vector1,vector2)

class greens_function:
    def __init__(self, verbose=0):
        self.verbose = verbose

    def solve_ip(self,cc,ps,qs,omega_list,broadening):

        eomip=pyscf.cc.eom_rccsd.EOMIP(cc)
        #cc=eom._cc
        ####
        cc.l2 = np.zeros_like(cc.l2)
        
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        if self.verbose > 0:
            print " solving ip portion..."
        x0 = initial_ip_guess(cc)
        p0 = 0.0*x0 + 1.0
        e_vector = list() 
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(cc,q))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=complex)
        for ip,p in enumerate(ps):
            b_vector = greens_b_vector_ip_rhf(cc,p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega-1j*broadening)
                solver = gminres.gMinRes(matr_multiply,b_vector,x0,p0)
                #solver = gminres.exactInverse(matr_multiply,b_vector,x0)
                sol = solver.get_solution().reshape(-1)
                x0  = sol
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iomega]  = -np.dot(e_vector[iq],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def solve_ea(self,cc,ps,qs,omega_list,broadening):
        eomea=pyscf.cc.eom_rccsd.EOMEA(cc)
        ####
        cc.l2 = np.zeros_like(cc.l2)

        
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        if self.verbose > 0:
            print " solving ea portion..."
        x0 = initial_ea_guess(cc)
        p0 = 0.0*x0 + 1.0
        e_vector = list() 
        for p in ps:
            e_vector.append(greens_e_vector_ea_rhf(cc,p))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=complex)
        for iq,q in enumerate(qs):
            b_vector = greens_b_vector_ea_rhf(cc,q)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega-1j*broadening)
                solver = gminres.gMinRes(matr_multiply,b_vector,x0,p0)
                #solver = gminres.exactInverse(matr_multiply,b_vector,x0)
                sol = solver.get_solution().reshape(-1)
                x0 = sol
                for ip,p in enumerate(ps):
                    gfvals[ip,iq,iomega] = np.dot(e_vector[ip],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def solve_gf(self,cc,p,q,omega_list,broadening):
        return self.solve_ip(cc,p,q,omega_list,broadening), self.solve_ea(cc,p,q,omega_list,broadening)

