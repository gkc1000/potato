import numpy as np
from potato import dmft


def get_hubbard_1d_ints(U, nimp, nbath, nkx=100):
    if nimp == 1:
        kx = np.arange(-nkx/2+1, nkx/2+1, dtype=float)
        hcore_k_ = -2*np.cos(2.*np.pi*kx/nkx)
        hcore_k  = hcore_k_.reshape([nkx,1,1])
        eri = np.zeros([1,1,1,1])
        eri[0,0,0,0] = U

    else:
        def nn_hopping(ns):
            t = np.zeros((ns,ns,), dtype=float)
            for ist in range(ns-1):
                t[ist,ist+1] = -1.0
                t[ist+1,ist] = -1.0
            t[0,-1] += -1.0
            t[-1,0] += -1.0
            return t

        def planewave(ns):
            U = np.zeros((ns,ns,), dtype=complex)
            scr = np.arange(ns, dtype=float)
            for k in range(ns):
                kk = (2.0*np.pi/ns)*k
                U[:,k] = np.exp(1j*kk*scr)
            U *= (1.0/np.sqrt(ns))
            return U

        nkx_ = nkx/nimp
        T = nn_hopping(nkx)
        Ut = planewave(nkx_)
        hcore_k = np.zeros((nkx_,nimp,nimp,), dtype=complex)
        for i1 in range(nimp):
            for i2 in range(nimp):
                for k in range(nkx_):
                    T_ = T[i1::nimp,i2::nimp].\
                         reshape((nkx_,nkx_,), order='F')
                    hcore_k[k,i1,i2] = \
                            np.dot(Ut[:,k].T, np.dot(T_, Ut[:,k].conj()))

        eri = np.zeros([nimp,nimp,nimp,nimp])
        for p in range(nimp):
            eri[p,p,p,p] = U

    return hcore_k, eri


def dmft_hub_1d(U, nimp, nbath, mu, delta, solver_type='cc'):
    hcore_k, eri = get_hubbard_1d_ints(U, nimp, nbath)

    mydmft = dmft.DMFT(hcore_k, eri, nbath, solver_type=solver_type)
    mydmft.chkfile = None 
    mydmft.verbose = 7
    mydmft.diis = True
    mydmft.gmres_tol = 1e-3

    mydmft.kernel(mu0=mu, delta=delta, opt_mu=False)
    occupancy = np.trace(mydmft.get_rdm_imp())/(mydmft.nao)
    print "At mu =", mu, ", occupancy =", occupancy

    mydmft.verbose = 0
    mydmft._scf.mol.verbose = 0
    wl, wh = -10.+U/2., 10.+U/2.
    freqs = np.linspace(wl, wh, 64)
    # for plotting
    eta = 0.5
    ldos = mydmft.get_ldos_imp(freqs, eta)[0]

    filename = 'hubbard_1d_U-%.0f_mu-%0.2f_n-%0.2f_%d-%d_d-%.1f_wh-7_%s.dat'%(
                U,mu,occupancy,nimp,nbath,delta,solver_type)
    with open(filename, 'w') as f:
        f.write('# n = %0.12g\n'%(occupancy))
        for w,freq in enumerate(freqs):
            f.write('%0.12g %.12g %.12g\n'%(freq, freq-U/2., ldos[w]))

    omega_ns = np.linspace(0., 20., 64)[1:]
    sigma = mydmft.get_sigma_imp(1j*omega_ns, 0.0)[0,0]
    sigma = np.imag(sigma)
    filename = 'hubbard_1d_U-%.0f_mu-%0.2f_n-%0.2f_%d-%d_d-%.1f_wh-7_%s_sigma.dat'%(
                U,mu,occupancy,nimp,nbath,delta,solver_type)
    with open(filename, 'w') as f:
        for n,wn in enumerate(omega_ns):
            f.write('%.12g %.12g\n'%(wn, sigma[n]))

def main():
    U = 6.00
    nimp = 2
    # nbath sites per impurity
    nbath = 2 
    delta = 0.1

    # half filling
    mu = U/2.
    dmft_hub_1d(U,nimp,nbath,mu,delta,'cc')
    dmft_hub_1d(U,nimp,nbath,mu,delta,'fci')

    # doped
    mu = 0.
    dmft_hub_1d(U,nimp,nbath,mu,delta,'cc')
    dmft_hub_1d(U,nimp,nbath,mu,delta,'fci')


if __name__ == '__main__':
    main()
