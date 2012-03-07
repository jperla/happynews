#!/usr/bin/env python
"""
    Looks inside outputed model.
    Prints out topics and other variables.
    Useful for inspecting a model's fit.

    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import random
import sys
import jsondata

import numpy as np

if __name__=='__main__':
    phi_filename = sys.argv[1]
    vocab_filename = sys.argv[2]
    num_docs = int(sys.argv[3])
    eta_filename = sys.argv[4]

    associated_filename = sys.argv[5]

    phi = jsondata.read(phi_filename)
    lexicon = jsondata.read(vocab_filename)

    eta = None
    if eta_filename is not None:
        eta = jsondata.read(eta_filename)

    print 'eta: %s' % eta

    if associated_filename is not None:
        associated = jsondata.read(associated_filename)

    def predict(eta, phi):
        Ks = len(eta)
        N,K = phi.shape
        phi = phi[:,-Ks:]
        EZ = np.sum(phi, axis=0) / N
        return np.dot(eta, EZ)

    print 'read in data...'
    predicted_ratings = list(sorted((predict(eta, p),i) for i,p in enumerate(phi)))
    print 'predicted ratings...'

    print 'most positive...'
    p = predicted_ratings[:num_docs]
    random.shuffle(p)
    for predicted,d in p:
        #print predicted, associated[d]
        print '%s (%s)' % (associated[d][1], d)

    print 'negative...'
    p = predicted_ratings[-num_docs:]
    random.shuffle(p)
    for predicted,d in p:
        #print predicted, associated[d]
        print '%s (%s)' % (associated[d][1], d)



