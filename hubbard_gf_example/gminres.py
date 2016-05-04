import numpy as np
import sys

def setA(sizer, plus):
    A = np.zeros(shape=(sizer,sizer),dtype=complex)
    P = np.zeros(shape=(sizer,1),dtype=complex)
    fac =1.0
    for i in xrange(sizer):
      A[i,i] = 1.0 * fac + 6.0 * 1j * fac + plus + 30.*np.random.random()
      P[i]  = 1./( A[i,i] )
      #P[i] = 1. + 1j * 0.0
      if i+2 < sizer:
        A[i,i+2] = 1.0 * fac
      if i+3 < sizer:
        A[i,i+3] = 0.7 * fac
      if i+1 < sizer:
        A[i+1,i] = 3.0*1j * fac
    #P = np.conj(P)
    return A, P


class gMinRes:
    def __init__(self,inMatrMultiply,inBvec,xStart,inPreCon):
        self.matrMultiply=inMatrMultiply
        self.maxM=130
        self.size=xStart.shape[0]
        self.maxOuterLoop=10
        self.totaliteration=0
        self.tol=1e-6
        self.iteration=0
        self.beta=0.0
        self.converged=0
        self.previousDRes=0.0
        self.currentDRes=0.0
        self.dresidual=0.0
        self.ratioRes=999.9

        #
        #  Creating initial guess, b vector, and preconditioner
        #

        self.b=inBvec.copy()
        self.x0=xStart.copy().reshape(-1,1)
        #self.x0 *= 0.0
        #print "Starting with initial guess of 0..."

        self.preCon=1.0*inPreCon.copy() + 0.0j*np.ones(self.size)

        #
        #  Allocating other vectors and iterating
        #

        self.allocateVecs()
        while self.converged == 0:
            self.guessInitial()
            for i in xrange(self.maxM-1):
                self.gms()
                self.qr()
                self.checkDeflate()
                self.iteration += 1
                self.totaliteration += 1
                if self.converged:
                    break
        #print ""
        #print "**Converged in %3d cycles" % self.totaliteration

    def allocateVecs(self):
        self.subH=np.zeros( shape=(self.maxM,self.maxM-1), dtype=complex )
        self.dgks=np.zeros( shape=(self.maxM-1), dtype=complex )
        self.eb  =np.zeros( shape=(self.maxM,1), dtype=complex )
        self.Ax=np.zeros( shape=(self.size,1), dtype=complex )
        self.givensR=np.zeros( shape=(self.maxM-1,self.maxM-1), dtype=complex )
        self.givensC=np.zeros( shape=(self.maxM-1), dtype=complex )
        self.givensS=np.zeros( shape=(self.maxM-1), dtype=complex )
        self.sol=np.zeros( shape=(self.maxM-1), dtype=complex )
        self.r0=np.zeros( shape=(self.size,1), dtype=complex )
        self.vlist=np.zeros( shape=(self.maxM,self.size,1), dtype=complex )

    def getSolution(self):
        return self.finalSol

    def checkDeflate(self):
        if self.iteration == self.maxM-2: #deflate
            #print "deflating space..."
            #print ""
            #print ""
            #print ""
            self.x0 = self.xnoob
            self.subH *= 0.0
            self.givensR *= 0.0

    def guessInitial(self):
        args = 0
        self.preCon  = self.matrMultiply(np.ones(self.size),args) + 0.0j*np.ones(self.size)
        #self.preCon += np.ones(self.size) * (1j*eta + omega)
        Ax  = self.matrMultiply(self.x0.reshape(self.size),args)
        #Ax += self.x0.reshape(self.size) * (1j*eta + omega)
        Ax  = Ax.reshape( Ax.shape[0], 1 )

        #print "Ax0 "
        #for i in xrange(100):
        #    dval = Ax[i]
        #    if abs(dval) > 1e-14:
        #        print i, dval

        self.r0  = np.zeros( shape=(self.size,1), dtype=complex )
        self.r0 += self.b.reshape(self.size,1) - Ax
        self.r0  = np.divide(np.ravel(self.r0),np.ravel(self.preCon)).reshape(self.size,1)

        #print "r0 "
        #for i in xrange(100):
        #    dval = abs(self.r0[i])
        #    if abs(dval) > 1e-14:
        #        print i, dval

        #print "initial ax "
        #print Ax

        #print "initial r0 "
        #print r0

        self.beta=np.vdot(self.r0,self.r0) ** 0.5
        self.eb[0] = self.beta
        self.vlist[0]=self.r0/self.beta
        self.iteration=0
        #print " :: initial residual (beta) = ", self.beta

    def gms(self):
        args = 0
        self.previousDRes = self.currentDRes
        Ax = self.matrMultiply(self.vlist[self.iteration].flatten(),args)
        #Ax += self.vlist[self.iteration].reshape(self.size) * (1j*eta + omega )
        Ax = Ax.reshape( -1, 1 )
        Ax = np.divide(Ax.flatten(),self.preCon.flatten()).reshape(self.size,1)

        #print "Ax "
        #for i in xrange(100):
        #    dval = Ax[i]
        #    if abs(dval) > 1e-14:
        #        print i, dval

        #print "ax before gms ..."
        #print Ax
#
#  MGS
#
#        for i in xrange(self.iteration+1):
#            self.subH[i,self.iteration] = np.vdot( self.vlist[i], Ax )
#            Ax -= self.subH[i,self.iteration]*self.vlist[i]
#
#  CGS with DGKS correction
#
        for i in xrange(self.iteration+1):
            self.subH[i,self.iteration] = np.vdot( self.vlist[i], Ax )
        for i in xrange(self.iteration+1):
            Ax -= self.subH[i,self.iteration]*self.vlist[i]
        self.vlist[self.iteration+1]=Ax

        #print "Ax after orthogonalizing..."
        #for i in xrange(100):
        #    dval = Ax[i]
        #    if abs(dval) > 1e-14:
        #        print i, dval


        norm = np.vdot(Ax,Ax) ** 0.5
        #print " :: h[j+1,j] = ", norm
        #print ""
        #print "iteration :", self.iteration+1


        if norm < 1e-8:
            self.qr()
            return

        #self.checkOrthog()

#
# Do a DGKS correction if we have to
#
#        print "Performing DGKS ..."
        for k in xrange( 1 ):
            for i in xrange(self.iteration+1):
                #self.dgks[i] = np.vdot(self.vlist[i],self.vlist[self.iteration+1])
                self.dgks[i] = np.vdot(self.vlist[i],self.vlist[self.iteration+1])
                self.vlist[self.iteration+1] -= self.vlist[i]*self.dgks[i]
                self.subH[i,self.iteration] += (self.dgks[i])

            #self.checkOrthog()

        norm = np.vdot(self.vlist[self.iteration+1],self.vlist[self.iteration+1]) ** 0.5
        self.subH[self.iteration+1,self.iteration]=norm
        self.vlist[self.iteration+1]/=norm
        #print "NORM AFTER DGKS", norm
        self.checkOrthog()




    def checkOrthog(self):
        max_orthog = -999.9
        for i in xrange(self.iteration+2):
            orthog = 0.0
            if i < self.iteration+1:
                orthog=np.real( np.vdot(self.vlist[self.iteration+1],self.vlist[i]) )
                orthog = abs(orthog) ** 0.5
                if orthog > max_orthog:
                    max_orthog = orthog
            if i == self.iteration+1:
                orthog=np.real( np.vdot(self.vlist[i],self.vlist[self.iteration+1]) )
                orthog = abs(orthog) ** 0.5
                if max_orthog > 1e-7:
                    print ""
                    print ""
                    print "*****************"
                    print "  MAX ORTHOG = ", max_orthog
                    print "  VECNORM = ", orthog
                    print "*****************"
                    print ""
                    print ""


    def qr(self):
        j=self.iteration+1
        noob=self.subH[:j+1,:j]
        ebnew=np.zeros( shape=(self.maxM,1), dtype=complex )
        ebnew[0]=self.beta
        ebnew=ebnew[:j+1]
        solution=np.linalg.lstsq(noob,ebnew)[0]

        q,r = np.linalg.qr(noob)
        w, v = np.linalg.eig(noob[:j,:j])
        idx = w.real.argsort()[::1]
        w = w[idx]
        v = v[:,idx]

        xm = 0.0*self.x0.copy()
        for kk in range( 0, j ):
            xm += self.vlist[kk]*solution[kk]
        self.xnoob=xm
        self.xnoob+=self.x0

        args = 0
        temp  = self.matrMultiply(self.xnoob.reshape(self.size),args)
        #temp += self.xnoob.reshape(self.size) * (1j*eta + omega)
        vres  = self.b.reshape(self.size) - temp


        cresidual=np.vdot(vres,vres) ** 0.5
        self.currentDRes = abs(cresidual)
        self.dresidual = abs(self.previousDRes - self.currentDRes)
        self.ratioRes = abs(self.dresidual)/abs(self.currentDRes)
        #print self.xnoob
        #print "current residual = ", cresidual
        if cresidual < self.tol:
            self.finalSol = self.xnoob
            self.converged=1
            #print "FINAL SOLUTION = "
            #for i in xrange(self.size):
            #    dval = np.vdot(self.xnoob[i],self.xnoob[i])
            #    if dval > 1e-14:
            #        print "x%-5d :%20.16f %20.16f" % (i, np.real(self.xnoob[i]), np.imag(self.xnoob[i]))

    def get_solution(self):
        return self.xnoob


def main():
    size=300
    A,P=setA(size,0.0+1j*0.0)
    b = np.random.rand(size,1) + 0j*np.random.rand(size,1)
    b /= np.sqrt( np.vdot(b,b) )

    xstart = np.dot(np.linalg.inv(A),b)
    xstart += 1./1.*(np.random.rand(size,1) + 0j*np.random.rand(size,1))
    condition_number = np.linalg.cond(A)
    r0=b-np.dot(A,xstart)
    res=np.vdot(r0,r0) ** 0.5
    finalx=np.dot(np.linalg.inv(A),b)
    print " ::: Making A,b matrix :::"
    print "  - condition number = %12.8f" % condition_number
    print "  - x0 residual      = %12.8f" % np.real(res)
    #for i in xrange(size):
    #    print "%20.16f %20.16f" % (np.real(finalx[i]), np.imag(finalx[i]))

    def multiplyA(vector,args):
        return np.dot(A,vector).reshape(len(vector))
    def multiplyA_precon(vector,args):
        return np.multiply(P.reshape(len(vector)),np.dot(A,vector).reshape(len(vector))).reshape(len(vector))
    gmin = gMinRes(multiplyA,b,xstart,P)

    #b = np.multiply(b,P)
    #gmin = gMinRes(multiplyA_precon,b,xstart,P)
    sol = gmin.get_solution()

    print "|Ax-b| = ", np.linalg.norm(np.dot(A,sol) - b)


if __name__ == '__main__':
    main()
