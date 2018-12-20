#!/usr/bin/python

import numpy
from pyscf.lib import logger
import pyscf.scf as scf


class RHF(scf.hf.RHF):
    __doc__ = scf.hf.SCF.__doc__

    def __init__(self, mol, mu, smearing=None):
        self.mu = mu
        self.smearing = smearing
        scf.hf.SCF.__init__(self, mol)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        if self.smearing:
            for n,e in enumerate(mo_energy):
                mo_occ[n] = 2./(numpy.exp((e-self.mu)/self.smearing)+1)
        else:
            mo_occ[mo_energy<=self.mu] = 2.
        nmo = mo_energy.size
        nocc = int(numpy.sum(mo_occ) // 2)
        if self.verbose >= logger.INFO and nocc < nmo:
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])
            else:
                logger.info(self, '  nelec = %d', nocc*2)
                logger.info(self, '  HOMO = %.15g  LUMO = %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])

        if self.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(self, '  mo_energy =\n%s', mo_energy)
            numpy.set_printoptions(threshold=1000)
        return mo_occ

