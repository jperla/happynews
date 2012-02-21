#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Uses graphlib and topiclib to run the Latent Response Mood Model
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""

from itertools import izip,chain
from functools import partial

try:
    import numpypy as np
except ImportError:
    import numpy as np
    np.seterr(invalid='raise')


import graphlib
import topiclib


class LatentMoodVars(graphlib.GraphVars):
    """Algorithm:
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

    def __init__(self, data=None, K=None, J=None):
        self.documents = None
        self.comments = None
        self.alphaD = None
        self.alphaC = None
        self.betaD = None
        self.betaC = None
        self.gammaD = None
        self.gammaC = None
        self.phiD = None
        self.phiC = None
        self.y = None
        self.eta = None
        self.sigma_squared = None
        self.is_initialized = False

        if data is not None:
            self.set_documents(data)

        if K is not None:
            self.initialize(K, J)


    def set_documents(self, data):
        """Accepts a 2-tuple of arrays of dictionaries. 
        Each element in 2-tuple is an array of documents.
        Each document is a dictionary (sparse vector).
        Saves the data locally.  Computes vocabulary.
        """
        documents,comments = data
        assert len(documents) == len(comments)

        self.documents = documents
        self.comments = comments

        self.vocab = max(chain(*[[w[0] for w in d] for d in self.documents]))
        self.vocab = max(self.vocab, max(chain(*[[w[0] for w in c] for c in self.comments])))
        self.W = self.vocab + 1
        self.D = len(documents)

        self.optimize_documents()

    def iterdocs(self):
        """Documents are computed in E-step in turn. Yields generator of documents.
        In this case, 2-tuples of (document,comment).
        """
        return izip(self.documents, self.comments)

    def optimize_documents(self):
        """Converts the local documents from sparse representation into normal vector."""
        # OPTIMIZATION: turn all documents into arrays
        self.documents = [topiclib.doc_to_array(d) for d in self.documents]
        self.comments = [topiclib.doc_to_array(c) for c in self.comments]

    def initialize(self, K, J):
        """Accepts K number of topics in document, and J number of topics in comments.
            Initializes all of the hidden variable arrays now that it knows dimensions
            of topics, vocabulary, etc.
        """
        assert self.documents is not None
        assert 2 <= K <= J # want more topics in comments, maybe (and at least 2)

        # give at least more documents than topics
        # so that it's not singular
        assert self.D > (K+J)

        self.K = K
        self.J = J

        D = self.D
        W = self.W

        # "it suffices to fix alpha to uniform 1/K"
        # initialize to ones so that the topics are more evenly distributed
        # good for small datasets
        self.alphaD = np.ones((K,)) * (3.0 / K)
        self.alphaC = np.ones((J,)) * (3.0 / J)

        # Initialize the variational distribution q(beta|lambda)
        self.betaD = topiclib.initialize_beta(K, W)
        self.betaC = topiclib.initialize_beta(J, W)

        # calculate number of words per document/comment
        document_Nds = self.num_words_per(self.documents)
        comment_Nds = self.num_words_per(self.comments)
        self.phiD = [(np.ones((document_Nds[d], K))*(1.0/K)) for d in xrange(D)]
        self.phiC = [(np.ones((comment_Nds[d], J))*(1.0/J)) for d in xrange(D)]

        self.gammaD = np.ones((D, K)) * (1.0 / K)
        graphlib.initialize_random(self.gammaD)
        self.gammaC = np.ones((D, J)) * (1.0 / J)
        graphlib.initialize_random(self.gammaC)

        self.y = graphlib.random_normal(0.0, 2.0, (D,))
        print 'y start: {0}'.format(self.y)

        self.eta = graphlib.random_normal(0.0, 3.0, (K+J,))
        self.sigma_squared = 10.0

        # EXPERIMENT
        # Let's initialize eta so that it agrees with y at start
        # hopefully this will keep y closer to gaussian centered at 0
        topiclib.lm_recalculate_eta_sigma(self.eta, self.y, self.phiD, self.phiC)

        print 'eta start: {0}'.format(self.eta)

        self.is_initialized = True

    def to_dict(self):
        return { 'y': self.y, 'eta': self.eta, 'sigma_squared': self.sigma_squared,
                    'betaD': self.betaD, 'betaC': self.betaC,
                    'gammaD': self.gammaD, 'gammaC': self.gammaC,
                    'phiD': self.phiD, 'phiC': self.phiC, }


def lm_e_step(global_iterations, v):
    for d, (document,comment) in enumerate(v.iterdocs()):
        local_i = topiclib.lm_E_step_for_doc(global_iterations, d, 
                                        document, comment, 
                                        v.alphaD, v.alphaC, 
                                        v.betaD, v.betaC, 
                                        v.gammaD[d], v.gammaC[d], 
                                        v.phiD[d], v.phiC[d], 
                                        v.y, v.eta, v.sigma_squared)
    return local_i

def lm_m_step(var):
    ### M-step: ###
    print 'updating betas..'
    # update betaD for documents first
    topiclib.lda_recalculate_beta(var.documents, var.betaD, var.phiD)
    print 'comments..'
    # update betaC for comments next
    topiclib.lda_recalculate_beta(var.comments, var.betaC, var.phiC)

    print 'eta sigma...'
    # update response variable gaussian global parameters
    var.sigma_squared = topiclib.lm_recalculate_eta_sigma(var.eta, var.y, var.phiD, var.phiC)

lm_global_elbo = lambda v: topiclib.lm_global_elbo(v.documents, v.comments, v.alphaD, v.alphaC, v.betaD, v.betaC, v.gammaD, v.gammaC, v.phiD, v.phiC, v.y, v.eta, v.sigma_squared)

def lm_print_func(var):
    print 'y: %s' % var.y
    print 'eta: %s' % var.eta
    print 'ss: %s' % var.sigma_squared

run_latent_mood = partial(graphlib.run_variational_em, e_step_func=lm_e_step, 
                                                        m_step_func=lm_m_step, 
                                                        global_elbo_func=lm_global_elbo,
                                                        print_func=lm_print_func)


            
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

    import jsondata
    numdocs = 1000000
    limit_words = 50000000
    docs = jsondata.read('documents.dc.nyt.json')[:numdocs]
    docs = [d[:limit_words] for d in docs]
    comments = jsondata.read('comments.dc.nyt.json')[:numdocs]
    comments = [c[:limit_words] for c in comments]

    print '%s total words in documents' % sum(len(d) for d in docs)
    print '%s total words in comments' % sum(len(c) for c in comments)

    real_data = [docs, comments]

    
    var = LatentMoodVars(test_data, K=2, J=3)
    var = LatentMoodVars(noisy_test_data, K=2, J=3)
    var = LatentMoodVars(real_data, K=10, J=20)

    try:
        output = run_latent_mood(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

