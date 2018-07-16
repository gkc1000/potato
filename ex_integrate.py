import numpy as np
import scipy.linalg
import scipy.integrate

COUNTER = 0

def _tb(n):
    """
    Tight-binding Hamiltonian
    """
    h=np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            if abs(i-j)==1:
                h[i,j]=1.
    h[0,-1]=1.
    h[-1,0]=1.
    return h

def gf(w, delta, nao=4):
    e = np.array([-i for i in range(nao)])
    gf = np.diag(1./(w + 1j*delta - e))
    return gf

def sigma0(w, nao):
    """
    Model constant SE
    """
    s = 0.1 * np.eye(nao)
    return s

def sigma1(w, nao):
    """
    Model 1/w part of SE
    """
    s = np.zeros([nao,nao],np.complex128)
    for i in range(nao):
        s[i,i] = (1.+.1*1j)/w
    return s

def integrate():
    delta = 1.e-1
    mu = -0.1
    nao = 4
    
    def a_fn(w):
        return -1./np.pi * np.imag(np.trace(gf(w, delta)))

    print "dos at mu", np.imag(np.trace(gf(mu, delta)))
    print "real-axis", scipy.integrate.quad(a_fn,-6.,mu)

    def imag_fn(w):
        return -2./np.pi * np.real(np.trace(gf(1j*w+mu, delta)))

    assert nao == gf(0, delta).shape[0]

    # NL = # poles to left of mu, NR = # poles to right of mu
    # nao = NL + NR
    # integration gives NR - NL (factor of 2 in imag_fn)
    print "imag axis", .5 * (nao - (scipy.integrate.quad(imag_fn,0,100000)[0]))

    # energy due to a constant self-energy
    def e0_fn(w):
        return -1./np.pi * np.imag(np.trace(np.dot(gf(w,delta), sigma0(w, nao))))

    # energy due to 1/w self-energy
    def e1_fn(w):
        return -1./np.pi * np.imag(np.trace(np.dot(gf(w,delta), sigma1(w, nao))))

    # energy integration along real axis
    print "real axis, E0", scipy.integrate.quad(e0_fn,-8.,mu)
    print "real axis, E1", scipy.integrate.quad(e1_fn,-8.,mu)

    def imag_e0_fn(w):
        return -2./np.pi * np.real(np.trace(np.dot(gf(1j*w + mu,delta), sigma0(1j*w + mu, nao))))
                                    
    def imag_e1_fn(w):
        return -2./np.pi * np.real(np.trace(np.dot(gf(1j*w + mu,delta), sigma1(1j*w + mu, nao))))

    # energy due to constant self-energy
    # This can be obtained once the density matrix is computed
    print "imag axis E0", .5 * (np.trace(sigma0(mu, nao)) - scipy.integrate.quad(imag_e0_fn,0,10000)[0])
    
    # energy due to 1/w self-energy
    
    #print "imag axis E1", -.5 * scipy.integrate.quad(imag_e1_fn,0,10000)[0]
    stuff = scipy.integrate.quad(imag_e1_fn,0,10000,full_output=True)
    print -.5* stuff[0], stuff[2]["neval"]


