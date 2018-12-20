import time
import sys

import numpy as np
import scipy
import gmres

import pyscf
import pyscf.cc
from pyscf.lib import logger
from pyscf.cc.eom_rccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea

'''
CCSD Green's function
'''


def greens_b_singles_ea_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return -t1[p,:]
    else:
        p = p-nocc
        result = np.zeros((nvir,), dtype=ds_type)
        result[p] = 1.0
        return result


def greens_b_doubles_ea_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return -t2[p,:,:,:]
    else:
        return np.zeros((nocc,nvir,nvir), dtype=ds_type)


def greens_b_vector_ea_rhf(cc, p):
    return amplitudes_to_vector_ea(
        greens_b_singles_ea_rhf(cc.t1, p),
        greens_b_doubles_ea_rhf(cc.t2, p),
    )


def greens_e_singles_ea_rhf(t1, t2, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return l1[p, :]
    else:
        p = p-nocc
        result = np.zeros((nvir,), dtype=ds_type)
        result[p] = -1.0
        result += np.einsum('ia,i->a', l1, t1[:,p])
        result += 2*np.einsum('klca,klc->a', l2, t2[:,:,:,p])
        result -= np.einsum('klca,lkc->a', l2, t2[:,:,:,p])
        return result


def greens_e_doubles_ea_rhf(t1, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return 2*l2[p,:,:,:] - l2[:,p,:,:]
    else:
        p = p-nocc
        result = np.zeros((nocc,nvir,nvir), dtype=ds_type)
        result[:,p,:] += -2*l1
        result[:,:,p] += l1
        result += 2*np.einsum('k,jkba->jab', t1[:,p], l2)
        result -= np.einsum('k,jkab->jab', t1[:,p], l2)
        return result


def greens_e_vector_ea_rhf(cc, p):
    return amplitudes_to_vector_ea(
        greens_e_singles_ea_rhf(cc.t1, cc.t2, cc.l1, cc.l2, p),
        greens_e_doubles_ea_rhf(cc.t1, cc.l1, cc.l2, p),
    )


def greens_b_singles_ip_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = 1.0
        return result
    else:
        p = p-nocc
        return t1[:,p]


def greens_b_doubles_ip_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return np.zeros((nocc,nocc,nvir), dtype=ds_type)
    else:
        p = p-nocc
        return t2[:,:,p,:]


def greens_b_vector_ip_rhf(cc, p):
    return amplitudes_to_vector_ip(
        greens_b_singles_ip_rhf(cc.t1, p),
        greens_b_doubles_ip_rhf(cc.t2, p),
    )


def greens_e_singles_ip_rhf(t1, t2, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc,), dtype=ds_type)
        result[p] = -1.0
        result += np.einsum('ia,a->i', l1, t1[p,:])
        result += 2*np.einsum('ilcd,lcd->i', l2, t2[p,:,:,:])
        result -= np.einsum('ilcd,ldc->i', l2, t2[p,:,:,:])
        return result
    else:
        p = p-nocc
        return -l1[:,p]


def greens_e_doubles_ip_rhf(t1, l1, l2, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = np.zeros((nocc, nocc, nvir), dtype=ds_type)
        result[p, :, :] += -2*l1
        result[:, p, :] += l1
        result += 2*np.einsum('c,ijcb->ijb', t1[p,:], l2)
        result -= np.einsum('c,jicb->ijb', t1[p,:], l2)
        return result
    else:
        p = p-nocc
        return -2*l2[:,:,p,:] + l2[:,:,:,p]


def greens_e_vector_ip_rhf(cc, p):
    return amplitudes_to_vector_ip(
        greens_e_singles_ip_rhf(cc.t1, cc.t2, cc.l1, cc.l2, p),
        greens_e_doubles_ip_rhf(cc.t1, cc.l1, cc.l2, p),
    )


def greens_func_multiply(ham, vector, linear_part, **kwargs):
    return np.array(ham(vector, **kwargs) + linear_part * vector)


def ip_shape(cc):
    nocc, nvir = cc.t1.shape
    return nocc + nocc*nocc*nvir


def ea_shape(cc):
    nocc, nvir = cc.t1.shape
    return nvir + nocc*nvir*nvir


class CCGF(object): 
    def __init__(self, mycc, tol=1e-4, verbose=None):
        self._cc = mycc
        self.tol = tol
        if verbose:
            self.verbose = verbose
        else:
            self.verbose = self._cc.verbose
        self.stdout = sys.stdout

    def ipccsd_ao(self, ps, omega_list, mo_coeff, broadening):
        eomip = pyscf.cc.eom_rccsd.EOMIP(self._cc)
        eomip_imds = eomip.make_imds()
        diag = eomip.get_diag() 

        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ip_shape(self._cc)], dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo[i,:] = greens_e_vector_ip_rhf(self._cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps,:], e_vector_mo)
        b_vector_mo = np.zeros([ip_shape(self._cc), nmo], dtype=np.complex128)
        for i in range(nmo):
            b_vector_mo[:,i] = greens_b_vector_ip_rhf(self._cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:,ps])

        gf_ao = np.zeros((len(ps), len(ps), len(omega_list)), dtype=np.complex128)
        for ip, p in enumerate(ps):
            x0 = None
            for iomega in range(len(omega_list))[::-1]:
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                diag_w = diag + curr_omega-1j*broadening
                if x0 is None:
                    x0 = b_vector_ao[:,p]/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector_ao[:,p], x0, diag_w, tol=self.tol)
                cput1 = (time.clock(), time.time())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'IPGF GMRES orbital p = %d/%d, freq w = %d/%d (%d iterations)'%(
                    ip+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
                x0 = sol
                for iq, q in enumerate(ps):
                    gf_ao[ip,iq,iomega] = -np.dot(e_vector_ao[iq,:], sol)
        return gf_ao

    def eaccsd_ao(self, ps, omega_list, mo_coeff, broadening):
        eomea = pyscf.cc.eom_rccsd.EOMEA(self._cc)
        eomea_imds = eomea.make_imds()
        diag = eomea.get_diag() 

        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ea_shape(self._cc)], dtype=np.complex128)
        for i in range(nmo):
            e_vector_mo[i,:] = greens_e_vector_ea_rhf(self._cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps,:], e_vector_mo)
        b_vector_mo = np.zeros([ea_shape(self._cc), nmo], dtype=np.complex128)
        for i in range(nmo):
            b_vector_mo[:,i] = greens_b_vector_ea_rhf(self._cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:,ps])

        gf_ao = np.zeros((len(ps), len(ps), len(omega_list)), dtype=np.complex128)
        for iq, q in enumerate(ps):
            x0 = None
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                diag_w = diag + (-curr_omega-1j*broadening)
                if x0 is None:
                    x0 = b_vector_ao[:,q]/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector_ao[:,q], x0, diag_w, tol=self.tol)
                cput1 = (time.clock(), time.time())
                sol = solver.solve().reshape(-1)
                cput1 = logger.timer(self, 'EAGF GMRES orbital q = %d/%d, freq w = %d/%d (%d iterations)'%(
                    iq+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
                x0 = sol
                for ip, p in enumerate(ps):
                    gf_ao[ip,iq,iomega] = np.dot(e_vector_ao[ip,:], sol)

        return gf_ao

    def ipccsd_mo(self, ps, qs, omega_list, broadening):
        eomip = pyscf.cc.eom_rccsd.EOMIP(self._cc)
        eomip_imds = eomip.make_imds()
        diag = eomip.get_diag() 
        e_vector = list()
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(self._cc, q))
        gfvals = np.zeros((len(ps), len(qs), len(omega_list)), dtype=complex)
        for ip, p in enumerate(ps):
            b_vector = greens_b_vector_ip_rhf(self._cc, p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

                diag_w = diag + curr_omega-1j*broadening
                x0 = b_vector[:,p]/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector[:,p], x0, diag_w, tol=self.tol)
                sol = solver.solve().reshape(-1)
                x0 = sol
                for iq, q in enumerate(qs):
                    gfvals[ip,iq,iomega] = -np.dot(e_vector[iq], sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def eaccsd_mo(self, ps, qs, omega_list, broadening):
        eomea = pyscf.cc.eom_rccsd.EOMEA(self._cc)
        eomea_imds = eomea.make_imds()
        diag = eomea.get_diag() 
        e_vector = list()
        for p in ps:
            e_vector.append(greens_e_vector_ea_rhf(self._cc, p))
        gfvals = np.zeros((len(ps), len(qs), len(omega_list)), dtype=complex)
        for iq, q in enumerate(qs):
            b_vector = greens_b_vector_ea_rhf(self._cc, q)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]

                def matr_multiply(vector, args=None):
                    return greens_func_multiply(eomea.matvec, vector, -curr_omega - 1j * broadening, imds=eomea_imds)

                diag_w = diag + (-curr_omega-1j*broadening)
                x0 = b_vector[:,q]/diag_w
                solver = gmres.GMRES(matr_multiply, b_vector[:,q], x0, diag_w, tol=self.tol)
                sol = solver.solve().reshape(-1)
                x0 = sol
                for ip, p in enumerate(ps):
                    gfvals[ip,iq,iomega] = np.dot(e_vector[ip], sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def get_gf(self, p, q, omega_list, broadening):
        return (self.ipccsd_mo(p, q, omega_list, broadening), 
                self.eaccsd_mo(p, q, omega_list, broadening))


    # time-dependent CC, below, is a work in progress


    def td_ip_ao(self, cc, ps, times, mo_coeff, re_im="re", tol=1.e-5):
        """
        E0: total CC gs energy
        ti: initial time
        tf: final time
        times: list of times where GF is computed
        tol : rtol, atol for ODE integrator
                   
        mo_coeff : integrals are assumed in the MO basis
        they are supplied here so we can back transform
        to the AO basis

        re_im in {"re", "im"}

        Signs, etc. Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        re_im = "re" corresponds to G^<(it) in Eq. A.3, with t = times
        re_im = "im" corresponds to G^<(\tau) in Eq. A.2, with \tau = times        
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = 1
        elif re_im == "re":
            dtype = np.complex128
            tfac = 1j
        else:
            raise RuntimeError

        # set initial bra/ket
        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ip_shape(cc)], dtype=dtype)
        for i in range(nmo):
            e_vector_mo[i, :] = greens_e_vector_ip_rhf(cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps, :], e_vector_mo)

        b_vector_mo = np.zeros([ip_shape(cc), nmo], dtype=dtype)
        for i in range(nmo):
            b_vector_mo[:, i] = greens_b_vector_ip_rhf(cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:, ps])

        # initialize loop variables
        gf_ao = np.zeros((len(ps), len(ps), len(times)), dtype=dtype)
        eomip = pyscf.cc.eom_rccsd.EOMIP(cc)
        eomip_imds = eomip.make_imds()
        ti = times[0]
        tf = times[-1]

        def matr_multiply(t, vector):
            # note: t is a dummy time argument, H is time-independent
            return tfac * np.array(eomip.matvec(vector, imds=eomip_imds))

        for ip, p in enumerate(ps):
            solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector_ao[:, p], t_eval=times,
                                             rtol=tol, atol=tol)

            for iq, q in enumerate(ps):
                gf_ao[iq, ip, :] = np.dot(e_vector_ao[iq], solp.y)

        return gf_ao

    def td_ea_ao(self, cc, ps, times, mo_coeff, re_im="re", tol=1.e-5):
        """
        See td_ip.
        
        Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        corresponds to G^>(it) in Eq. A.3
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = -1
        elif re_im == "re":
            dtype = np.complex128
            tfac = -1j
        else:
            raise RuntimeError

        # set initial bra/ket
        nmo = mo_coeff.shape[1]
        e_vector_mo = np.zeros([nmo, ea_shape(cc)], dtype=dtype)
        for i in range(nmo):
            e_vector_mo[i, :] = greens_e_vector_ea_rhf(cc, i)
        e_vector_ao = np.einsum("pi,ix->px", mo_coeff[ps, :], e_vector_mo)

        b_vector_mo = np.zeros([ea_shape(cc), nmo], dtype=dtype)
        for i in range(nmo):
            b_vector_mo[:, i] = greens_b_vector_ea_rhf(cc, i)
        b_vector_ao = np.einsum("xi,ip->xp", b_vector_mo, mo_coeff.T[:, ps])

        # initialize loop variables
        gf_ao = np.zeros((len(ps), len(ps), len(times)), dtype=dtype)
        eomea = pyscf.cc.eom_rccsd.EOMEA(cc)
        eomea_imds = eomea.make_imds()
        ti = times[0]
        tf = times[-1]

        def matr_multiply(t, vector):
            # note: t is a dummy time argument, H is time-independent
            return tfac * np.array(eomea.matvec(vector, imds=eomea_imds))

        for ip, p in enumerate(ps):
            solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector_ao[:, p], t_eval=times,
                                             rtol=tol, atol=tol)

            for iq, q in enumerate(ps):
                gf_ao[iq, ip, :] = np.dot(e_vector_ao[iq], solp.y)

        return gf_ao

    def td_ip(self, cc, ps, qs, times, re_im="re", tol=1.e-5):
        """
        E0: total CC gs energy
        ti: initial time
        tf: final time
        times: list of times where GF is computed
        tol : rtol, atol for ODE integrator
                   
        re_im in {"re", "im"}

        Signs, etc. Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        re_im = "re" corresponds to G^<(it) in Eq. A.3, with t = times
        re_im = "im" corresponds to G^<(\tau) in Eq. A.2, with \tau = times        
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = 1
        elif re_im == "re":
            dtype = np.complex128
            tfac = 1j
        else:
            raise RuntimeError

        eomip = pyscf.cc.eom_rccsd.EOMIP(cc)
        eomip_imds = eomip.make_imds()
        ti = times[0]
        tf = times[-1]

        e_vector = list()
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(cc, q))

        gfvals = np.zeros((len(ps), len(qs), len(times)), dtype=dtype)

        for ip, p in enumerate(ps):
            b_vector = np.array(greens_b_vector_ip_rhf(cc, p), dtype=dtype)

            def matr_multiply(t, vector, args=None):
                # note: t is a dummy time argument, H is time-independent
                res = tfac * np.array(eomip.matvec(vector, imds=eomip_imds))
                return res

            solp = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector, t_eval=times, rtol=tol, atol=tol)

            for iq, q in enumerate(qs):
                gfvals[iq, ip, :] = np.dot(e_vector[iq], solp.y)

        return gfvals

    def td_ea(self, cc, ps, qs, times, re_im="re", tol=1.e-5):
        """
        See td_ip.

        Defn. at https://edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf, pg. 141
        corresponds to G^>(it) in Eq. A.3
        """
        dtype = None
        tfac = None
        if re_im == "im":
            dtype = np.float64
            tfac = -1
        elif re_im == "re":
            dtype = np.complex128
            tfac = -1j
        else:
            raise RuntimeError

        ti = times[0]
        tf = times[-1]
        eomea = pyscf.cc.eom_rccsd.EOMEA(cc)
        eomea_imds = eomea.make_imds()

        e_vector = list()
        for p in ps:
            e_vector.append(np.array(greens_e_vector_ea_rhf(cc, p), dtype=dtype))
        gfvals = np.zeros((len(ps), len(qs), len(times)), dtype=complex)

        for iq, q in enumerate(qs):
            b_vector = np.array(greens_b_vector_ea_rhf(cc, q), dtype=dtype)

            def matr_multiply(t, vector, args=None):
                # t is a dummy time argument
                res = tfac * np.array(eomea.matvec(vector, imds=eomea_imds))
                return res

            solq = scipy.integrate.solve_ivp(matr_multiply, (ti, tf),
                                             b_vector, t_eval=times, rtol=tol, atol=tol)

            for ip, p in enumerate(ps):
                gfvals[ip, iq, :] = np.dot(e_vector[ip], solq.y)
        return gfvals

