class DMFT(object):
    def __init__(self):
        pass

def kernel(dmft, hcore_kpts, eri, freqs, wts, delta, conv_tol):
    """
    DMFT self-consistency

    Modeled after PySCF HF kernel
    """
    dmft_conv = False
    cycle = 0

    # get initial guess
    hyb = get_hyb(hcore_kpts, np.zeros_like(hcore_kpts[0,:,:]),
                  freqs, delta)
    
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb
        sigma = get_sigma(hcore_kpts, eri, hyb, freqs, wts, delta)
        hyb = get_hyb(hcore_kpts, sigma, freqs, delta)
        norm_hyb = numpy.linalg.norm(hyb-hyb_last)

        # this would be a good place to put DIIS
        # or damping
        
        if (norm_hyb < conv_tol):
            dmft_conv = True

        cycle +=1

    return hyb, sigma
    
def get_bath(hyb, freqs, wts):
    """
    Convert hybridization function 
    to bath couplings and energies

    Args:
        hyb : (nimp, nimp, nw) ndarray
        freqs : (nw) ndarray
        wts : (nw) ndarray, Gaussian wts at freq pts

    Returns:
        bath_v : (nimp, nimp*nw) ndarray
        bath_e : (nimp*nw) ndarray
    """
    nw = len(freqs)
    wh = max(freqs)
    wl = min(freqs)

    dw = (wh - wl) / (nw - 1)
    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb)

    # simple discretization of bath, Eq. (9), arxiv:1507.07468
    v = np.empty_like(v2)

    for iw in range(nw):
        eig, vec = scipy.linalg.eigh(v2[:,:,iw])

        # although eigs should be positive, there
        # could be numerical-zero negative eigs: check this
        neg_eig = [e for e in eig if e < 0]
        assert np.allclose(neg_eig, 0)

        v[:,:,iw] = np.dot(vec, np.diag(np.abs(eig))**0.5) * math.sqrt(wts[iw])

    nimp = hyb.shape[0]
    bath_v = np.reshape(v, [nimp, -1])
    bath_e = np.zeros([nimp * nw])

    # bath_e is [nimp * nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for ip in range(nimp):
        for iw in range(nw):
            bath_e[ip*nw + iw] = freqs[iw]

    return bath_v, bath_e

def get_sigma(hcore_kpts, eri, hyb, freqs, wts, delta):
    """
    Impurity self energy
    """
    hcore_cell = 1./nkpts * numpy.sum(hcore_kpts, axis=0)

    # bath representation of hybridization
    bath_v, bath_e = get_bath(hyb, freqs, wts)
    nbath = len(bath_e)
    himp = np.zeros([nao + nbath, nao + nbath])
    himp[:nao,:nao] = hcore_cell
    himp[:nao, nao:] = bath_v
    himp[nao:, :nao] = bath_v.T
    himp[nao:,nao:] = np.diag(bath_e)

    gf_imp = get_interacting_gf(himp, eri, freqs, delta) # This is the impurity solver
    gf0_imp = get_gf(himp, np.zeros_like(himp), freqs, delta)[:nao,:nao]
    
    sigma = np.zeros_like(gf_imp)
    for iw in range(nw):
        sigma[:,:,iw] = inv(gf0_imp[:,:,iw]) - inv(gf_imp[:,:,iw])
    return sigma
    
def get_hyb(hcore_kpts, sigma, freqs, delta):
    """
    Hybridization
    """
    nkpts = hcore_kpts.shape[0]
    nao = hcore_kpts.shape[1]
    nw = len(freqs)
    
    gf_kpts = get_gf_kpts(hcore_kpts, sigma, freqs, delta)
    gf_cell = 1./nkpts * numpy.sum(gf_kpts, axis=0)
    hcore_cell = 1./nkpts * numpy.sum(hcore_kpts, axis=0)

    gf0_cell = get_gf(hcore_cell, sigma, freqs, delta)

    hyb = np.zeros_like(gf0_imp)
    for iw in range(nw):
        hyb[:,:,iw] = inv(gf0_cell[:,:,iw]) - inv(gf_cell[:,:,iw])

    return hyb

def get_gf_kpts(hcore_kpts, sigma, freqs, delta):
    """
    kpt Green's function at a set of frequencies

    Args: 
         hcore_kpts : (nkpts, nao, nao) ndarray
         sigma : (nao, nao) ndarray
         freqs : (nw) ndarray
         delta : float

    Returns:
         gf_kpts : (nkpts, nao, nao) ndarray
    """
    nkpts = hcore_kpts.shape[0]
    nao = hcore_kpts.shape[1]
    gf_kpts = np.zeros([nkpts, nao, nao, nw], np.complex128)

    for k in range(nkpts):
        gf_kpts[k,:,:,:] = get_gf(hcore_kpts[k,:,:], sigma, delta)
    return gf_kpts
    
def get_gf(hcore, sigma, freqs, delta):
    """
    Green's function at a set of frequencies

    Args: 
         hcore : (nao, nao) ndarray
         sigma : (nao, nao) ndarray
         freqs : (nw) ndarray
         delta : float

    Returns:
         gf : (nao, nao) ndarray

    """
    nao = hcore.shape[0]
    nw  = len(freqs)
    gf = np.zeros([nao, nao, nw], np.complex128)
    for iw, w in enumerate(freqs):
        gf[:,:,iw] = inv((w+1j*delta)*np.eye(nao)-hcore-sigma)
    return gf
