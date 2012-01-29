#!/usr/bin/env python
import os
from itertools import cycle, repeat

import numpy as np

import graphlib
import topiclib

Ku = 21
Ks = 5
Kb = 20

D = 80
L = 81
B = 82

W = 1000 # vocab size

# todo: change this to find errors
N = 90 # every document has 100 words


K = Ku + Ks + Kb

eta = ([0.0] * Ku) + [e[1] for e in zip(range(Ks), cycle((+2.0,-2.0,-1.0,1.0)))] + ([0.0] * Kb)
sigma_squared = 0.01

multiplier = 1.0

# the magical priors that separate and share topics
alphaU = multiplier * np.array(([1.0 / Ku] * Ku) + ([0.0] * (Ks + Kb)))
alphaS = multiplier * np.array(([0.0] * Ku) + ([1.0 / Ks] * Ks) + ([0.0] * Kb))
alphaB = multiplier * np.array(([0.0] * (Ku + Ks)) + ([1.0 / Kb] * Kb))

# generate topics for unlabeled documents 
# (note that alphaD is a D-length list of alphas)
alphaD = [np.array(np.random.dirichlet(alphaU)) for d in xrange(D)]
thetaD = [np.random.dirichlet(alphaD[d]) for d in xrange(D)]
thetaC = [np.random.dirichlet(alphaD[d] + alphaS) for d in xrange(D)]

# generate topics for labeled comments
thetaL = [np.random.dirichlet(alphaS + alphaB) for l in xrange(L)]
# generate topics for background distribution of labeled comments
thetaB = [np.random.dirichlet(alphaB) for b in xrange(B)]



# now, generate the z's from multinomials
# todo: check that thetas are probabilities
zD = [np.random.multinomial(N, thetaD[d]) for d in xrange(D)]
zC = [np.random.multinomial(N, thetaC[d]) for d in xrange(D)]

zL = [np.random.multinomial(N, thetaL[l]) for l in xrange(L)]
zB = [np.random.multinomial(N, thetaB[b]) for b in xrange(B)]


def zbar(z):
    """Accepts a topic-length array of number of topics chosen for each document.
        Normalizes them to be a probability of choosing each document.
        Just divide by sum.
       Returns that.
    """
    z = np.array(z)
    return 1.0 * z / np.sum(z)

# figure out the ratings of the documents
muU = [np.dot(eta, zbar(zC[d])) for d in xrange(D)]
muL = [np.dot(eta, zbar(zL[l])) for l in xrange(L)]

yU = [np.random.normal(muU[d], np.sqrt(sigma_squared)) for d in xrange(D)]
yL = [np.random.normal(muL[l], np.sqrt(sigma_squared)) for l in xrange(L)]

# and finally, generate the word distribution beta and actual words
beta = topiclib.initialize_beta(K, W)

def itertopics(zvals):
    """Accepts a bunch of z values (a list of ints, number of times topic t is repeated).
        Yields a generator which repeats those values the appropriate times.
        Will yield sum(zvals) times.
    """
    for t,n in enumerate(zvals):
        for topic in repeat(t, n):
            yield topic

def make_doc(topics, beta):
    return np.sum([np.random.multinomial(1, beta[k]) for k in itertopics(topics)], axis=0)

documents = [make_doc(zD[d], beta) for d in xrange(D)]
comments = [make_doc(zC[d], beta) for d in xrange(D)]
labeled = [make_doc(zL[l], beta) for l in xrange(L)]
background = [make_doc(zB[b], beta) for b in xrange(B)]


# save the documents in sparse format
def save_sparse(filename, docs):
    """Accepts filename, and a list of documents in non-sparse form.
        Each doc is a list of integers.
        Saves them to a file in a line-by-line doc-by-doc wordid:count format.
    """
    sparse_docs = []
    for d in docs:
        counts = dict((w, n) for w,n in enumerate(d) if n > 0)
        sparse_docs.append([(k,v) for k,v in sorted(counts.iteritems())])
    with open(filename, 'w') as f:
        for d in sparse_docs:
            for w,n in d:
                f.write('{0}:{1} '.format(w,n))
            f.write('\n')

dirname = 'synthmodel'

save_sparse(os.path.join(dirname, 'documents.dat'), documents)
save_sparse(os.path.join(dirname, 'comments.dat'), comments)
save_sparse(os.path.join(dirname,'labeled.dat'), labeled)
save_sparse(os.path.join(dirname, 'background.dat'), background)

def save_in_dir(dirname, var_names, var_dict):
    for name in var_names:
        v = np.array(var_dict[name])
        fname = os.path.join(dirname, name + '.npy')
        np.savetxt(fname, v)

# save the hidden variables
save_in_dir(dirname, ['eta', 
                        'alphaU', 'alphaD', 'alphaS', 'alphaB',
                        'thetaD', 'thetaC', 'thetaL', 'thetaB',
                        'zD', 'zC', 'zL', 'zB',
                        'muU', 'muL',
                        'yU', 'yL',
                        'beta',
                        'documents', 'comments', 'labeled', 'background',
                      ], locals())


