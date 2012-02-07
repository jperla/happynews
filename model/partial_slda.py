#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Uses graphlib and topiclib to run partial sLDA
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""

global final_output

from itertools import chain,izip
from functools import partial

try:
    import numpy as np
    np.seterr(invalid='raise')
except:
    import numpypy as np


import graphlib
import topiclib


class PartialSupervisedLDAVars(graphlib.GraphVars):
    """
    Same as Supervisd LDA, 
        but only use Ks of the topics to multiply by eta to get y, the response variable.
    """

    def __init__(self, data=None, Ks=None, Kb=None):
        self.documents = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.phi = None
        self.eta = None
        self.sigma_squared = None
        self.is_initialized = False

        if data is not None:
            self.set_documents(data)

        if Ks is not None:
            self.initialize(Ks, Kb)

    def set_documents(self, data):
        """Accepts a 2-tuple of arrays of dictionaries. 
        Each element in 2-tuple is an array of documents.
        Each document is a dictionary (sparse vector).
        Saves the data locally.  Computes vocabulary.
        """
        documents,y = data
        assert len(documents) == len(y)

        self.documents = documents
        self.y = np.array(y)
        assert len(self.y.shape) == 1

        self.vocab = max(chain(*[[w[0] for w in d] for d in self.documents]))
        self.W = self.vocab + 1
        self.D = len(documents)

        self.optimize_documents()

    def iterdocs(self):
        """Documents are computed in E-step in turn. Yields generator of documents.
        In this case, 2-tuples of (document,comment).
        """
        return izip(self.documents, self.y)

    def optimize_documents(self):
        """Converts the local documents from sparse representation into normal vector."""
        # OPTIMIZATION: turn all documents into arrays
        self.documents = [topiclib.doc_to_array(d) for d in self.documents]

    def initialize(self, Ks, Kb):
        """Accepts K number of topics in document.
            Initializes all of the hidden variable arrays now that it knows dimensions
            of topics, vocabulary, etc.
        """
        assert self.documents is not None
        assert Ks is not None
        assert Kb is not None

        K = Ks + Kb

        # give at least more documents than topics
        # so that it's not singular
        assert self.D > K

        self.K = K

        D = self.D
        W = self.W

        # "it suffices to fix alpha to uniform 1/K"
        # initialize to ones so that the topics are more evenly distributed
        # good for small datasets
        self.alpha = np.ones((K,)) * (1.0 / K)

        # Initialize the variational distribution q(beta|lambda)
        self.beta = topiclib.initialize_beta(K, W)

        document_Nds = self.num_words_per(self.documents)
        self.phi = [(np.ones((document_Nds[d], K))*(1.0/K)) for d in xrange(D)]

        self.gamma = np.ones((D, K)) * (1.0 / K)
        graphlib.initialize_random(self.gamma)

        self.eta = graphlib.random_normal(0, 2.0, (Ks,))
        self.sigma_squared = 0.5

        print 'eta start: {0}'.format(self.eta)

        self.is_initialized = True

    def to_dict(self):
        return { 'eta': self.eta, 'sigma_squared': self.sigma_squared,
                 'beta': self.beta, 'gamma': self.gamma, 'phi': self.phi, }

def partial_slda_e_step(global_iterations, v):
    local_i = 0
    for d, (document,y) in enumerate(v.iterdocs()):
        local_i = topiclib.partial_slda_E_step_for_doc(global_iterations,
                                                        local_i,
                                                        d, document, y,
                                                        v.alpha, v.beta,
                                                        v.gamma[d], v.phi[d],
                                                        v.eta, v.sigma_squared)
    return local_i

def partial_slda_m_step(var):
    ### M-step: ###
    print 'updating betas..'
    # update betaD for documents first
    topiclib.lda_recalculate_beta(var.documents, var.beta, var.phi)

    print 'eta sigma...'
    # update response variable gaussian global parameters
    var.sigma_squared = topiclib.partial_slda_recalculate_eta_sigma(var.eta, var.y, var.phi)


import slda

run_partial_slda = partial(graphlib.run_variational_em, 
                            e_step_func=partial_slda_e_step, 
                            m_step_func=partial_slda_m_step, 
                            global_elbo_func=topiclib.partial_slda_global_elbo,
                            print_func=slda.slda_print_func)


            
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
                 1.7,
                 2.0,
                 1.2,
                 4.8,
                 5,
                 4.2
                ])
    test_data = (
                [
                 [(0,1), (2,2), (3,1), (4,1),],
                 [(0,1), (2,1), (3,2), (4,3),],
                 [(0,1), (2,3), (3,3), (4,1),],
                 [(5,1), (6,2), (8,1), (9,3),],
                 [(5,1), (6,2), (8,1), (9,1),],
                 [(5,2), (6,1), (8,1), (9,1),],
                ],
                [
                 1.7,
                 2.0,
                 1.2,
                 4.8,
                 5,
                 4.2,
                ])

    
    #var = SupervisedLDAVars(test_data, K=3)
    #var = SupervisedLDAVars(noisy_test_data, K=3)



    # use my big generated dataset
    labeled_documents = topiclib.read_sparse('data/synthbigtlc/labeled.dat')[:100]
    y = np.loadtxt('data/synthbigtlc/yL.npy')[:100]
    real_data = (labeled_documents, y)

    var = PartialSupervisedLDAVars(real_data, Ks=5, Kb=20)

    try:
        output = run_partial_slda(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

