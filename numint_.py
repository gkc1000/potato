import numpy as np
import scipy
import scipy.integrate


def int_quad_real (gf_fn, delta=1.e-5, \
                   epsabs=1.49e-08, epsrel=1.49e-08):
    assert (callable(gf_fn))

    def real_fn(w):
        return -1./np.pi * np.imag(gf_fn(w, delta))

    int_ = scipy.integrate.quad(real_fn, 0, +np.inf, epsabs=epsabs, \
                                epsrel=epsrel, full_output=1)
    print 'neval = ', int_[2]['neval']
    print 'integral = ', int_[0]
    return int_[0]


def int_quad_imag (gf_fn, mu, delta=1.e-5, \
                   epsabs=1.49e-08, epsrel=1.49e-08):
    assert (callable(gf_fn))

    def imag_fn(w):
        return -2./np.pi * np.real(gf_fn(1j*w+mu, delta))

    int_ = scipy.integrate.quad(imag_fn, 0, +np.inf, epsabs=epsabs, \
                                epsrel=epsrel, full_output=1)
    print 'neval = ', int_[2]['neval']
    print 'integral = ', int_[0]
    return int_[0]

