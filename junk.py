    hyb2 = np.zeros_like(hyb)
    # check if integration really works

    nimp = hyb.shape[0]
    delta = .5 * (wh-wl) / nw
    for iw, w in enumerate(freqs):
        for iw2, w2 in enumerate(freqs):
            for p in range(nimp):
                for q in range(nimp):
                    hyb2[p,q,iw] += v2[p,q,iw2] / (w-w2+1j*delta)

    print "J2 function"
    for iw, w in enumerate(freqs): 
        print hyb2[0, 0, iw], hyb[0, 0, iw]
        #print hyb2[0, 1, iw], hyb[0, 1, iw]
    print np.linalg.norm(hyb2 - hyb)

    hyb3 = make_hyb(bath_v, bath_e, freqs, delta)
    
    hyb4 = np.zeros_like(hyb)
    
    for iw, w in enumerate(freqs):
        for iw2, w2 in enumerate(freqs):
            for p in range(nimp):
                for q in range(nimp):
                    for i in range(nimp):
                        hyb4[p,q,iw] += np.conj(v[p,i,iw2]) * v[q,i,iw2] / (w-w2+1j*delta)


    for iw, w in enumerate(freqs): 
        print "(%10.6f) (%10.6f) (%10.6f) (%10.6f)" % (hyb2[0, 0, iw], hyb[0, 0, iw], hyb3[0, 0, iw], hyb4[0,0,iw])

