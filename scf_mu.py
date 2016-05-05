#!/usr/bin/python

import numpy
import pyscf.scf as scf

class RHF(scf.hf.RHF):
    __doc__ = scf.hf.SCF.__doc__

    def __init__ (self, mol, mu):
        self.mu = mu
        scf.hf.SCF.__init__ (self, mol)

    def get_occ (self, mo_energy=None, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[mo_energy<=self.mu] = 2.
        return mo_occ

