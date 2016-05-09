import numpy as np
import numpy.polynomial
import scipy.integrate

def _get_scaled_legendre_roots(wl, wh, nw):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [wl, wh]

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs, wts = numpy.polynomial.legendre.leggauss(nw)
    freqs +=1 
    freqs *= (wh - wl) / 2.
    freqs += wl 
    wts *= (wh - wl) / 2.

    print "sum", np.sum(wts)
    
    return freqs, wts

def _get_linear_freqs(wl, wh, nw):
    freqs = np.linspace(wl, wh, nw) 
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)
    return freqs, wts


def _get_imag_integration_freqs(wreal, wcut, winf, ngauss=10):
    """
    Frequencies to use along imaginary axis

    Structure of frequencies
              ngauss           1
    freqs [ gauss freqs ...  winf ]

    Returns:
        freqs: (ngauss + 2,) ndarray, all imag frequencies
        wts: (ngauss, ) ndarray, GL weights
    """
    eps = 1.e-12
    # gfreqs, wts = np.array(_get_scaled_legendre_roots(eps, wcut, ngauss), np.complex128)
    gfreqs, wts = np.array(_get_linear_freqs(eps, wcut, ngauss), np.complex128)
    gfreqs *= 1j

    gfreqs += wreal

    freqs = gfreqs.copy()
    #freqs = np.hstack([gfreqs, np.array([winf])])
    return freqs, wts
    
def int_gf(op, gf, freqs, analytic_freqs, wts, ngauss):
    """
    Integral of a frequency independent operator
    with the Green's function 
               
    Args:
        op : (nao, nao) ndarray
        gf : (nao, nao, nw) ndarray
        freqs : (nw,) ndarray
        wts : (ngauss,) ndarray
        ngauss : int, # Gaussian integration pts

    If gamma_1 is the 1PDM, then

    Returns:
        tr op gamma_1
    """
    # Numerically integrate up to ngauss freqs
    #ogf_num = -2./np.pi * np.einsum("ijw,w->ijw", gf[:,:,:ngauss], wts)
    ogf_num = -2./np.pi * np.einsum("ijw,w->ijw", gf[:,:,:ngauss], wts)
    ogf_num = np.real(np.einsum("ij,jkw->ik", op, ogf_num))

    # print "gauss int", np.trace(ogf_num)
    # # int
    # n = 0.
    # for iw in range(gf.shape[2]):
    #     n += -2./np.pi * np.real(np.trace((gf[:,:,iw]))) * wts[iw]
    # print "num int", n

    # Analytically integrate
    # Real part of the gf, on the imag axis, decays like 1/w^2
    nao = op.shape[0]
    ogf_analytic = np.zeros_like(ogf_num, np.complex128)

    ogf = np.einsum("ij,jkw->ikw", op, gf)
    
    # for p in range(nao):
    #     for q in range(nao):
    #         c = np.real(gf[p,q,-1]) * (freqs[-1]**2) # extracts const in the 1/w^2 term
    #         ogf_analytic[p,q] = -2./np.pi * c / freqs[-1] # int_freqs[ngauss]^inf c/w^2 dw

    ogf_analytic = -2./np.pi * imag_int_re_analytic_fn(np.real(ogf[:,:,ngauss:]), analytic_freqs)
    
    print "analytic part", np.trace(ogf_analytic)
    # Circle at infinity
    #ogf_circle = op

    print "components",  np.trace(op),  np.trace(ogf_num)
    # return .5 *(op - ogf_num - ogf_analytic)
    
    return .5 *(op - ogf_num)

def imag_int_re_analytic_fn(re_analytic, analytic_freqs):
    """
    Fit fn to inverse polynomials and integrate along +ve imag. axis
    Fn is assumed to be *real*, frequencies are ordered
    """
    assert np.linalg.norm(np.imag(re_analytic)) < 1.e-12
    
    inv_freqs = 1./analytic_freqs

    nao = re_analytic.shape[0]
    nw = len(analytic_freqs)
    # c0 + c1/w + c2/w^2 + c3/w^3 + ...
    # c0, c1 should be 0
    deg = nw

    int_fn = np.zeros([nao,nao])

    lower_freq = analytic_freqs[0]
    for p in range(nao):
        for q in range(nao):
            gf_poly=numpy.polynomial.polynomial.polyfit(inv_freqs, np.real(re_analytic[p,q,:]), nw)
            print "gf_poly", gf_poly
            # first two coefficients should be close to 0
            for deg in range(2, nw):
                int_fn[p,q] += _int_invw(deg, lower_freq)

    return int_fn

def _int_invw(n, wl):
    """
    int_wl^inf w^(-n)
    """
    assert n > 1
    return wl**(-(n-1)) / (-(n-1))

def int_energy(sigma, gf, freqs, wts, ngauss):
    """
    Integral of Green's function and self-energy
    """
    # Int with constant part of self-energy (from highest freq)
    sigma_inf = sigma[:,:,-1]

    gf_sigma_inf = int_gf(sigma_const, gf, freqs, wts, ngauss)
    
    # Int numerically
    gf_sigma_num = -2./np.pi * np.einsum("ijw,w->ijw", gf[:,:,:ngauss], wts)
    gf_sigma_num = np.real(np.einsum("ij,jkw->ik", op, ogf_num))

    # Analytically integrate
    gf_sigma_analytic = np.zeros_like(gf_sigma_num)

    gf_sigma_large_w = np.dot(gf[:,:,ngauss], sigma[:,:,ngauss])
    
    for p in range(nao):
        for q in range(nao):
            c = np.real(gf_sigma_large_w[p,q]) * (freqs[ngauss]**2) # extracts const in the 1/w^2 term
            gf_sigma_analytic[p,q] = 2./np.pi * c / freqs[ngauss] # int_freqs[ngauss-1]^inf c/w^2 dw

    return gf_sigma_inf + gf_sigma_num + gf_sigma_analytic


### TEST FNS ###

def gf0(w, delta, nao=1):
    e = np.array([-i for i in range(nao)])*1.

    gf = np.diag(1./(w + 1j*delta - e))
    return gf


def test():

    winf = 1000000
    fac = 10
    wcut = 100.
    nao = 1
    delta = 1.e-5
    mu = -2.5

    wreal = mu
    #freqs, wts = _get_imag_integration_freqs(wreal, wcut, winf, ngauss)


    
    npts = 128+1
    ngauss = npts

    freqs, wts = _get_linear_freqs(0., wcut, npts)

    nw_analytic = 10
    analytic_freqs = np.array([freqs[-1] * 2**i for i in range(nw_analytic)])

    all_freqs = np.hstack([freqs, analytic_freqs])
    
    nw = len(all_freqs)
    gfw = np.zeros([nao, nao, nw])

    trgf = np.zeros([len(all_freqs)], np.complex128)
    for iw, w in enumerate(all_freqs):
        print w
        gfw[:,:,iw] = gf0(1j*w + mu, delta)
        trgf[iw] = np.trace(np.real(gf0(1j*w + mu, delta)))
        
    print "romberg",  -2./np.pi * scipy.integrate.romb(trgf[:ngauss], dx=wcut/(npts-1))
    # print "simple", np.sum(trgf) * -2./np.pi * wcut/(npts-1)
    
    gf_int = int_gf(np.eye(nao), gfw, freqs, analytic_freqs, wts, ngauss)
    print gf_int


    # spectral fn helper
    def a_fn(w):
        return -1./np.pi * np.imag(np.trace(gf0(w, delta)))

    def imag_fn(w):
        return -2./np.pi * np.real(np.trace(gf0(1j*w+mu, delta)))

    #print "frequencies", freqs
    #print freqs[ngauss]
    
    # gfreqs, wts = np.array(_get_linear_freqs(0., wcut, ngauss), np.complex128)
    # gfreqs *= 1j
    # gfreqs += wreal

    # print len(wts), len(gfreqs)
    
    # n = 0.
    # for iw in range(gfw.shape[2]):
    #     n += imag_fn(freqs[iw])
    #     #-2./np.pi * np.real(np.trace((gfw[:,:,iw]))) * wts[iw]
    # print "num int", n

    
    print "real-axis", scipy.integrate.quad(a_fn,-6.,mu)  
    print "imag up to wcut", scipy.integrate.quad(imag_fn, 0, wcut)
    print "imag wcut to inf", scipy.integrate.quad(imag_fn, wcut, 10000)
  
    #print "imag-axis", np.trace(gf_int)
