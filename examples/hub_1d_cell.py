import numpy as np
from potato import dmft

def hub_1d_cell(U, nimp, nbath, nkx=100, fill=1., chkf=None, \
                solver_type='scf'):

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

    mydmft = dmft.DMFT(hcore_k, eri, nbath,
                       solver_type=solver_type)
    mydmft.chkfile = chkf
    mydmft.verbose = 7
    mydmft.diis = False
    #mydmft.damp = 0.7
    mydmft.conv_tol = 1e-2
    mydmft.sigma = np.empty([nimp,nimp,nbath])
    for iw in range(nbath):
        mydmft.sigma[:,:,iw] = U/2.*np.eye(nimp)

    delta = 0.1
    mu0 = U/2.
    mydmft.kernel(mu0=mu0, fill=fill, delta=delta, opt_mu=False)

    wl, wh = -7.+mydmft.mu, 7.+mydmft.mu
    freqs = np.linspace(wl, wh, 64)
    ldos = mydmft.get_ldos_imp(freqs, 0.5)[0]

    filename = 'hubbard_1d_U-%.0f_%d-%d_d-%.1f_wh-7.dat'%(U,nimp,nbath,delta)
    with open(filename, 'w') as f:
        for w,freq in enumerate(freqs):
            f.write('%.12g %.12g\n'%(freq-mydmft.mu, ldos[w]))

    omega_ns = np.linspace(0., 20., 64)
    sigma = mydmft.get_sigma_imp(1j*omega_ns, 0.0)[0,0]
    sigma = np.imag(sigma)
    filename = 'hubbard_1d_U-%.0f_%d-%d_d-%.1f_wh-7_sigma.dat'%(U,nimp,nbath,delta)
    with open(filename, 'w') as f:
        for n,wn in enumerate(omega_ns):
            f.write('%.12g %.12g\n'%(wn, sigma[n]))


if __name__ == '__main__':
    U = 4.0
    nimp = 2
    nbath = 9
    nkx = 100
    fill = 1.
    hub_1d_cell(U, nimp, nbath, nkx=nkx, fill=fill, solver_type='cc')
