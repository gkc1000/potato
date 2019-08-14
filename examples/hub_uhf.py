import numpy as np
import scipy
import dmft_uhf as dmft

def get_hubbard_1d_ints(U, nimp, nkx=100):
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


def get_hubbard_2d_ints(U, nx, ny, nkx=10, nky=10):
    if nx == ny == 1:
        kx = np.arange(-nkx/2+1, nkx/2+1, dtype=float)
        ky = np.arange(-nky/2+1, nky/2+1, dtype=float)
        kx_, ky_ = np.meshgrid(kx,ky)
        hcore_k_ = -2*np.cos(2.*np.pi*kx_.flatten(order='C')/nkx) \
                   -2*np.cos(2.*np.pi*ky_.flatten(order='C')/nky)
        hcore_k  = hcore_k_.reshape([nkx*nky,1,1])
        eri = np.zeros([1,1,1,1])
        eri[0,0,0,0] = U

    else:
        assert (nkx % nx == 0)
        assert (nky % ny == 0)

        def nn_hopping(nkx, nky):
            t = np.zeros((nky,nkx,nky,nkx,), dtype=float)
            for istx in range(nkx):
                for isty in range(nky-1):
                    t[isty,istx,isty+1,istx] = -1.0
                    t[isty+1,istx,isty,istx] = -1.0
                t[0,istx,-1,istx] += -1.0
                t[-1,istx,0,istx] += -1.0
            for isty in range(nky):
                for istx in range(nkx-1):
                    t[isty,istx,isty,istx+1] = -1.0
                    t[isty,istx+1,isty,istx] = -1.0
                t[isty,0,isty,-1] += -1.0
                t[isty,-1,isty,0] += -1.0
            return t

        def planewave(nkx, nky):
            ns = nkx*nky
            U = np.zeros((ns,ns,), dtype=complex)
            sx = np.arange(nkx, dtype=float)
            sy = np.arange(nky, dtype=float)
            scrx, scry = np.meshgrid(sx,sy, indexing='ij')
            scrx = scrx.reshape((ns,))
            scry = scry.reshape((ns,))
            k = 0
            for kx in range(nkx):
                kkx = (2.0*np.pi/nkx)*kx
                for ky in range(nky):
                    kky = (2.0*np.pi/nky)*ky
                    U[:,k] = np.exp(1j*(kkx*scrx+kky*scry))
                    k += 1
            U *= (1.0/np.sqrt(ns))
            return U

        nkx_ = nkx/nx
        nky_ = nky/ny
        ns_ = nkx_*nky_
        T = nn_hopping(nkx, nky)
        Ut = planewave(nkx_, nky_)

        hcore_k = np.zeros((ns_,ny,nx,ny,nx,), dtype=complex)
        for i1y in range(ny):
            for i1x in range(nx):
                for i2y in range(ny):
                    for i2x in range(nx):
                        T_ = T[i1y::ny,i1x::nx,i2y::ny,i2x::nx].\
                             reshape((ns_,ns_,), order='F')
                        for k in range(ns_):
                            hcore_k[k,i1y,i1x,i2y,i2x] = \
                                np.dot(Ut[:,k].T, np.dot(T_, Ut[:,k].conj()))
        hcore_k = hcore_k.reshape((ns_,ny*nx,ny*nx,), order='F')

        hcore_k_ = np.zeros((ns_,nx*ny))
        for k in range(ns_):
            hcore_k_[k,:] = scipy.linalg.eigh(hcore_k[k,:,:], \
                                              eigvals_only=True)

        eri = np.zeros([nx*ny,nx*ny,nx*ny,nx*ny])
        for p in range(nx*ny):
            eri[p,p,p,p] = U

    return hcore_k, eri


def dmft_hub(U, nimp, nbath, mu, delta, solver_type='ucc'):
    try:
        dimension = len(nimp)
    except TypeError:
        dimension = 1

    if dimension == 1:
        hcore_k, eri = get_hubbard_1d_ints(U, nimp)
    elif dimension == 2:
        nx, ny = nimp
        hcore_k, eri = get_hubbard_2d_ints(U, nx, ny)

    mydmft = dmft.DMFT(hcore_k, eri, nbath, solver_type=solver_type)
    mydmft.chkfile = None 
    mydmft.verbose = 5
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
    ldos = mydmft.get_ldos_imp(freqs, eta)[:,0,:]

    if dimension == 1:
        filename = 'hubbard_1d_U-%.0f_mu-%0.2f_n-%0.2f_%d-%d_d-%.1f_wh-7_%s'%(
                    U,mu,occupancy,nimp,nbath,delta,solver_type)
    elif dimension == 2:
        filename = 'hubbard_2d_U-%.0f_mu-%0.2f_n-%0.2f_%dx%d-%d_d-%.1f_wh-7_%s'%(
                    U,mu,occupancy,nx,ny,nbath,delta,solver_type)

    with open(filename+'_alpha.dat', 'w') as f:
        f.write('# n = %0.12g\n'%(occupancy))
        for w,freq in enumerate(freqs):
            f.write('%0.12g %.12g %.12g\n'%(freq, freq-U/2., ldos[0][w]/2.))

    with open(filename+'_beta.dat', 'w') as f:
        f.write('# n = %0.12g\n'%(occupancy))
        for w,freq in enumerate(freqs):
            f.write('%0.12g %.12g %.12g\n'%(freq, freq-U/2., ldos[1][w]/2.))

    omega_ns = np.linspace(0., 20., 64)[1:]
    sigma = mydmft.get_sigma_imp(1j*omega_ns, 0.0)[:,0,0,:]
    sigma = np.imag(sigma)
    with open(filename+'_sigma_alpha.dat', 'w') as f:
        for n,wn in enumerate(omega_ns):
            f.write('%.12g %.12g\n'%(wn, sigma[0][n]/2.))
    with open(filename+'_sigma_beta.dat', 'w') as f:
        for n,wn in enumerate(omega_ns):
            f.write('%.12g %.12g\n'%(wn, sigma[1][n]/2.))


def main():
    delta = 0.1

    ### 2D Hubbard model
    U = 2.0

    ## 2x2-site embedding
    nx = ny = 2
    # nbath sites per impurity
    nbath = 10

    # half filling
    mu = U/2.
    dmft_hub(U,(nx,ny),nbath,mu,delta,'ucc')
    exit()
    # doped
    mu = 0.
    dmft_hub(U,(nx,ny),nbath,mu,delta,'ucc')


if __name__ == '__main__':
    main()
