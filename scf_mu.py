#!/usr/bin/python

import numpy
import pyscf.scf as scf

from pyscf.lib import logger

class RHF(scf.hf.RHF):
    __doc__ = scf.hf.SCF.__doc__

    def __init__ (self, mol, mu):
        self.mu = mu
        scf.hf.SCF.__init__ (self, mol)

    def make_rdm1 (self, mo_coeff=None, mo_occ=None):
        mo_energy = self.mo_energy
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[mo_energy<=self.mu] = 2.
        self.mo_occ = mo_occ
        return scf.hf.make_rdm1(mo_coeff, mo_occ)

    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        self.mo_energy, self.mo_coeff = self.eig(h1e, s1e)
        # mo_energy, mo_coeff = self.eig(h1e, s1e)
        # mo_occ = self.get_occ(mo_energy, mo_coeff)
        # return self.make_rdm1(mo_coeff, mo_occ)
        return self.make_rdm1(self.mo_coeff)

