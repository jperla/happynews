#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from itertools import izip,chain,repeat

try:
    import numpy as np
    np.seterr(invalid='raise')
except:
    import numpypy as np

try:
    from scipy.special import psi, gammaln
except:
    from scipypy import psi, gammaln
    

import graphlib


INITIAL_ELBO = float('-inf')

final_output = {}


def run_em(data, K, J):
    """
    Algorithm:
    Initialize each βC and βD randomly.
    Repeat until the ELBO converges:
    For each document d:
        For the document and then the comment, do the E-step:
            Initialize φd,i = 1/K (or 1/J for the comment)
            Repeat until the local ELBO converges:
                Update γd 
                Update φd,i 
        Update response variable y
    For each topic j and k, update βjC and βkD 
    Update the response variable parameters η and σ2

    K: fixed number of topics in the document
    J: fixed number of topics in the comment

    D: number of documents
    W: number of words in vocabulary

    alpha: dirichlet hyperparameter (Kx1)
    beta: topic word distributions (KxW)
    phi: variational parameter to word hidden topic assignment Z (dictionary of (Ndx1) (number of words per document))
    gamma: variational parameter to document topic distributions theta (D x K matrix)


    y: real-valued response variable per document
    eta: mean of gaussian generating y
    sigma_squared: variance of gaussian generating y

    document_Nds: vector of number of words per document (Dx1)
    comment_Nds: vector of number of words per comment (Dx1)
    """
    documents,comments = data
    assert len(documents) == len(comments)

    vocab = max(chain(*[[w[0] for w in d] for d in documents]))
    vocab = max(vocab, max(chain(*[[w[0] for w in c] for c in comments])))
    W = vocab+1
    D = len(documents)

    # give at least more documents than topics
    # so that it's not singular
    assert D > (K+J)

    # calculate number of words per document/comment
    document_Nds = [sum(w[1] for w in d) for d in documents]
    comment_Nds = [sum(w[1] for w in c) for c in comments]

    # "it suffices to fix alpha to uniform 1/K"
    # initialize to ones so that the topics are more evenly distributed
    # good for small datasets
    alphaD = np.ones((K,)) * (3.0 / K)
    alphaC = np.ones((J,)) * (3.0 / J)

    # Initialize the variational distribution q(beta|lambda)
    betaD = graphlib.initialize_beta(K, W)
    betaC = graphlib.initialize_beta(J, W)

    phiD = [(np.ones((document_Nds[d], K))*(1.0/K)) for d in xrange(D)]
    phiC = [(np.ones((comment_Nds[d], J))*(1.0/J)) for d in xrange(D)]

    gammaD = np.ones((D, K)) * (1.0 / K)
    graphlib.initialize_random(gammaD)
    gammaC = np.ones((D, J)) * (1.0 / J)
    graphlib.initialize_random(gammaC)

    def random_normal(mu, sigma, shape):
        """Define my own random normal, since numpypy does not have np.random.normal ."""
        size = shape[0]
        n = np.array([random.gauss(mu, sigma) for i in xrange(size)])
        return n

    y = random_normal(0.0, 2.0, (D,))
    print 'y start: {0}'.format(y)

    eta = random_normal(0.0, 3.0, (K+J,))
    sigma_squared = 10.0

    # EXPERIMENT
    # Let's initialize eta so that it agrees with y at start
    # hopefully this will keep y closer to gaussian centered at 0
    graphlib.recalculate_eta_sigma(eta, y, phiD, phiC)

    print 'eta start: {0}'.format(eta)


    # OPTIMIZATION: turn all documents into arrays
    documents = [graphlib.doc_to_array(d) for d in documents]
    comments = [graphlib.doc_to_array(c) for c in comments]

    iterations = 0
    elbo = INITIAL_ELBO
    last_elbo = INITIAL_ELBO - 100
    local_i = 0
    #for globaliternum in xrange(100):
    while graphlib.elbo_did_not_converge(elbo, last_elbo, 
                                         iterations, criterion=0.1, max_iter=20):
        
        ### E-step ###
        for d, (document, comment) in enumerate(izip(documents,comments)):
            local_i = graphlib.do_E_step(iterations, d, document, comment, alphaD, alphaC, betaD, betaC, gammaD[d], gammaC[d], phiD[d], phiC[d], y, eta, sigma_squared)


        ### M-step: ###
        print 'updating betas..'
        # update betaD for documents first
        graphlib.recalculate_beta(documents, betaD, phiD)
        print 'comments..'
        # update betaC for comments next
        graphlib.recalculate_beta(comments, betaC, phiC)

        print 'eta sigma...'
        # update response variable gaussian global parameters
        sigma_squared = graphlib.recalculate_eta_sigma(eta, y, phiD, phiC)

        print 'will calculate elbo...'
        last_elbo = elbo
        elbo = graphlib.calculate_global_elbo(documents, comments, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y, eta, sigma_squared)

        # todo: maybe write all these vars every iteration (or every 10) ?

        iterations += 1

        final_output.update({'iterations': iterations,
                             'elbo': elbo,
                             'y': y,
                             'eta': eta, 'sigma_squared': sigma_squared,
                             'betaD': betaD,
                             'betaC': betaC,
                             'gammaD': gammaD,
                             'gammaC': gammaC,
                             'phiD': phiD,
                             'phiC': phiC, })
        #print final_output
        print 'y: %s' % y
        print 'eta: %s' % eta
        print 'ss: %s' % sigma_squared

        print '{1} ({2} per doc) GLOBAL ELBO: {0}'.format(elbo, iterations, local_i)

    return final_output


            
if __name__=='__main__':
    # documents are 2-tuples of document, comment
    noisy_test_data = ([
                 [(1,1), (2,1), (3,3), (5,2),], 
                 [(0,1), (2,3), (3,1), (4,1),],
                 [(1,2), (2,1), (4,2), (5,4),],
                 [(5,1), (6,4), (7,1), (9,1),],
                 [(5,2), (6,1), (7,2), (9,4),],
                 [(5,1), (6,2), (7,2), (8,1),],
                ],[
                 [(0,3), (2,1), (3,1), (7,1),],
                 [(0,1), (1,1), (3,4), (5,4),],
                 [(0,1), (2,5), (3,1), (4,1),],
                 [(5,2), (6,1), (8,2), (9,1),],
                 [(3,3), (5,1), (8,2), (9,2),],
                 [(0,2), (6,1), (7,1), (8,1),],
                ])
    test_data = ([
                 [(0,1), (2,2), (3,1), (4,1),],
                 [(0,1), (2,1), (3,2), (4,3),],
                 [(0,1), (2,3), (3,3), (4,1),],
                 [(5,1), (6,2), (8,1), (9,3),],
                 [(5,1), (6,2), (8,1), (9,1),],
                 [(5,2), (6,1), (8,1), (9,1),],
                 ],
                 [
                 [(0,2), (1,1), (3,3), (4,1),],
                 [(0,1), (1,2), (3,2), (4,1),],
                 [(0,1), (1,1), (3,2), (4,2),],
                 [(5,3), (6,1), (7,2), (9,1),],
                 [(5,1), (6,2), (7,1), (9,1),],
                 [(5,1), (6,1), (7,1), (9,2),],
                ])

    '''
    import jsondata
    numdocs = 100
    limit_words = 500
    docs = jsondata.read('documents.dc.nyt.json')[:numdocs]
    docs = [d[:limit_words] for d in docs]
    comments = jsondata.read('comments.dc.nyt.json')[:numdocs]
    comments = [c[:limit_words] for c in comments]

    print '%s total words in documents' % sum(len(d) for d in docs)
    print '%s total words in comments' % sum(len(c) for c in comments)

    real_data = [docs, comments]
    '''
    real_data = test_data

    try:
        output = run_em(real_data, 2, 3)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

