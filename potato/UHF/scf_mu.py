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

class UHF(scf.uhf.UHF):
    __doc__ = scf.uhf.UHF.__doc__

    def __init__(self, mol, mu, h1e_spin, smearing=None):
        self.mu = mu
        self.smearing = smearing
        self.h1e_spin = h1e_spin
        scf.uhf.UHF.__init__(self, mol)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        if self.smearing:
            for i in range(2):
                for n,e in enumerate(mo_energy[i]):
                    mo_occ[i][n] = 1./(numpy.exp((e-self.mu)/self.smearing)+1)
        else:
            for i in range(2):
                mo_occ[i][mo_energy[i]<=self.mu] = 1.
        nmo = mo_energy[0].size
        nocca = int(numpy.sum(mo_occ[0]))
        noccb = int(numpy.sum(mo_occ[1]))

        if self.verbose >= logger.INFO and nocca < nmo and noccb > 0 and noccb < nmo:
            if mo_energy[0][nocca-1]+1e-3 > mo_energy[0][nocca]:
                logger.warn(self, 'alpha HOMO %.15g == LUMO %.15g',
                            mo_energy[0][nocca-1], mo_energy[0][nocca])
            else:
                logger.info(self, '  alpha nelec = %d', nocca)
                logger.info(self, '  alpha HOMO = %.15g  LUMO = %.15g',
                            mo_energy[0][nocca-1], mo_energy[0][nocca])

            if mo_energy[1][noccb-1]+1e-3 > mo_energy[1][noccb]:
                logger.warn(self, 'beta HOMO %.15g == LUMO %.15g',
                            mo_energy[1][noccb-1], mo_energy[1][noccb])
            else:
                logger.info(self, '  beta nelec = %d', noccb)
                logger.info(self, '  beta HOMO = %.15g  LUMO = %.15g',
                            mo_energy[1][noccb-1], mo_energy[1][noccb])

        if self.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(self, '  mo_energy =\n%s', mo_energy)
            numpy.set_printoptions(threshold=1000)

        return mo_occ

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        h1e = self.h1e_spin.copy()
        #if h1e is None: h1e = self.h1e_spin.copy()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        f = h1e + vhf
        if f.ndim == 2:
            f = (f, f)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp
        if s1e is None: s1e = self.get_ovlp()
        if dm is None: dm = self.make_rdm1()

        if isinstance(level_shift_factor, (tuple, list, numpy.ndarray)):
            shifta, shiftb = level_shift_factor
        else:
            shifta = shiftb = level_shift_factor
        if isinstance(damp_factor, (tuple, list, numpy.ndarray)):
            dampa, dampb = damp_factor
        else:
            dampa = dampb = damp_factor

        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = [dm*.5] * 2
        if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
            f = (hf.damping(s1e, dm[0], f[0], dampa),
                 hf.damping(s1e, dm[1], f[1], dampb))
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm, f, self, h1e, vhf)
        if abs(shifta)+abs(shiftb) > 1e-4:
            f = (hf.level_shift(s1e, dm[0], f[0], shifta),
                 hf.level_shift(s1e, dm[1], f[1], shiftb))
        return numpy.array(f)

