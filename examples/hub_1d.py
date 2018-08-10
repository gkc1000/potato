import numpy as np
from potato import dmft

def hub_1d(U, nbath, nkx=2, fill=1., chkf=None, \
           solver_type='scf'):
    kx = np.arange(-nkx/2+1, nkx/2+1, dtype=float)
    hcore_k_ = -2*np.cos(2.*np.pi*kx/nkx)
    hcore_k  = hcore_k_.reshape([nkx,1,1])
    eri = np.zeros([1,1,1,1])
    eri[0,0,0,0] = U

    mydmft = dmft.DMFT(hcore_k, eri, nbath,
                       solver_type=solver_type)
    mydmft.chkfile = chkf
    mydmft.verbose = 7
    mydmft.diis = False
    #mydmft.damp = 0.7
    mydmft.sigma = U/2.*np.ones([1,1,nbath])

    delta = 0.1
    mu0 = U/2.
    mydmft.kernel(mu0=mu0, fill=fill, delta=delta, opt_mu=False)

    wl, wh = -7.+mydmft.mu, 7.+mydmft.mu
    freqs = np.linspace(wl, wh, 64)
    ldos = mydmft.get_ldos_imp(freqs, 0.5)[0]

    filename = 'hubbard_1d_U-%.0f_%d-%d_d-%.1f_wh-7.dat'%(U,1,nbath,delta)
    with open(filename, 'w') as f:
        for w,freq in enumerate(freqs):
            f.write('%.12g %.12g\n'%(freq-mydmft.mu, ldos[w]))

    omega_ns = np.linspace(0., 20., 64)
    sigma = mydmft.get_sigma_imp(1j*omega_ns, 0.0)[0,0]
    sigma = np.imag(sigma)
    filename = 'hubbard_1d_U-%.0f_%d-%d_d-%.1f_wh-7_sigma.dat'%(U,1,nbath,delta)
    with open(filename, 'w') as f:
        for n,wn in enumerate(omega_ns):
            f.write('%.12g %.12g\n'%(wn, sigma[n]))


if __name__ == '__main__':
    U = 4.0
    nbath = 9 
    nkx = 100
    fill = 1.
    hub_1d(U, nbath, nkx=nkx, fill=fill, solver_type='cc')
