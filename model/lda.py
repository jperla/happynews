#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
from functools import partial

try:
    import numpy as np
    np.seterr(invalid='raise')
except:
    import numpypy as np


import graphlib
import topiclib


class LDAVars(graphlib.GraphVars):
    """Algorithm:
    Initialize β randomly.
    Repeat until the ELBO converges:
        For each document d:
            For the document, do the E-step:
                Initialize φd,i = 1/K (or 1/J for the comment)
                Repeat until the local ELBO converges:
                    Update φd,i 
            Update response variable y
        For each topic k, update βk

    K: fixed number of topics in the document

    D: number of documents
    W: number of words in vocabulary

    alpha: dirichlet hyperparameter (Kx1)
    beta: topic word distributions (KxW)
    phi: variational parameter to word hidden topic assignment Z (dictionary of (Ndx1) (number of words per document))
    gamma: variational parameter to document topic distributions theta (D x K matrix)
    """

    def __init__(self, documents=None, K=None):
        self.documents = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.phi = None
        self.is_initialized = False

        if documents is not None:
            self.set_documents(documents)

        if K is not None:
            self.initialize(K)

    def set_documents(self, data):
        """Accepts a 2-tuple of arrays of dictionaries. 
        Each element in 2-tuple is an array of documents.
        Each document is a dictionary (sparse vector).
        Saves the data locally.  Computes vocabulary.
        """
        documents = data
        self.documents = documents

        self.vocab = max(chain(*[[w[0] for w in d] for d in self.documents]))
        self.W = self.vocab + 1
        self.D = len(documents)

        self.optimize_documents()

    def iterdocs(self):
        """Documents are computed in E-step in turn. Yields generator of documents.
        In this case, 2-tuples of (document,comment).
        """
        return self.documents

    def optimize_documents(self):
        """Converts the local documents from sparse representation into normal vector."""
        # OPTIMIZATION: turn all documents into arrays
        self.documents = [topiclib.doc_to_array(d) for d in self.documents]

    def initialize(self, K):
        """Accepts K number of topics in document.
            Initializes all of the hidden variable arrays now that it knows dimensions
            of topics, vocabulary, etc.
        """
        assert self.documents is not None

        # give at least more documents than topics
        # so that it's not singular
        assert self.D > K

        self.K = K

        D = self.D
        W = self.W

        # "it suffices to fix alpha to uniform 1/K"
        # initialize to ones so that the topics are more evenly distributed
        # good for small datasets
        self.alpha = np.ones((K,)) * (3.0 / K)

        # Initialize the variational distribution q(beta|lambda)
        self.beta = topiclib.initialize_beta(K, W)

        document_Nds = self.num_words_per(self.documents)
        self.phi = [(np.ones((document_Nds[d], K))*(1.0/K)) for d in xrange(D)]

        self.gamma = np.ones((D, K)) * (1.0 / K)
        graphlib.initialize_random(self.gamma)

        self.is_initialized = True

    def to_dict(self):
        return { 'beta': self.beta, 'gamma': self.gamma, 'phi': self.phi, }

def lda_e_step(global_iterations, v):
    for d, document in enumerate(v.iterdocs()):
        local_i = topiclib.lda_E_step_for_doc(global_iterations, 
                                                d, document,
                                                v.alpha, v.beta,
                                                v.gamma[d], v.phi[d])
    return local_i

def lda_m_step(var):
    print 'updating betas..'
    topiclib.lda_recalculate_beta(var.documents, var.beta, var.phi)

def lda_print_func(var):
    #print 'phi: %s' % var.phi
    print 'gamma: %s' % var.gamma

run_lda = partial(graphlib.run_variational_em, e_step_func=lda_e_step, 
                                                    m_step_func=lda_m_step, 
                                                    global_elbo_func=topiclib.lda_global_elbo,
                                                    print_func=lda_print_func)


            
if __name__=='__main__':
    # documents are 2-tuples of document, comment
    noisy_test_data = [
                 [(1,1), (2,1), (3,3), (5,2),], 
                 [(0,1), (2,3), (3,1), (4,1),],
                 [(1,2), (2,1), (4,2), (5,4),],
                 [(5,1), (6,4), (7,1), (9,1),],
                 [(5,2), (6,1), (7,2), (9,4),],
                 [(5,1), (6,2), (7,2), (8,1),],
                ]
    test_data = [
                 [(0,1), (2,2), (3,1), (4,1),],
                 [(0,1), (2,1), (3,2), (4,3),],
                 [(0,1), (2,3), (3,3), (4,1),],
                 [(5,1), (6,2), (8,1), (9,3),],
                 [(5,1), (6,2), (8,1), (9,1),],
                 [(5,2), (6,1), (8,1), (9,1),],
                ]

    
    var = LDAVars(test_data, K=3)
    var = LDAVars(noisy_test_data, K=3)
    #var = LDAVars(real_data, K=20)

    try:
        output = run_lda(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

