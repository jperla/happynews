#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Uses graphlib and topiclib to run TLC model
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""

global final_output

import jsondata

from itertools import chain,izip
from functools import partial

try:
    import numpy as np
    np.seterr(invalid='raise')
except:
    import numpypy as np


import graphlib
import topiclib


class TLCVars(graphlib.GraphVars):
    """
    Transfer Learning C

    Mostly it is a partial Supervised LDA.  
    It also shares strength with other distributions to discover the background topics.
    This focuses the response variable to be conditions from just a few topics.
    The rest are used as background.
    """

    def __init__(self, data=None, Ku=None, Ks=None, Kb=None):
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

        if Ku is not None:
            self.initialize(Ku, Ks, Kb)

    def set_documents(self, data):
        """Accepts a 2-tuple of arrays of dictionaries. 
        Each element in 2-tuple is an array of documents.
        Each document is a dictionary (sparse vector).
        Saves the data locally.  Computes vocabulary.
        """
        documents,comments,labeled,background,y = data
        assert len(documents) == len(comments)
        assert len(labeled) == len(y)

        self.documents = documents
        self.comments = comments
        self.labeled = labeled
        self.background = background

        self.y = np.array(y)
        assert len(self.y.shape) == 1

        def max_word_in_sparse_text(text, previous_max):
            m = max(chain(*[[w[0] for w in d] for d in text]))
            return max(m, previous_max)

        self.vocab = max_word_in_sparse_text(self.documents, 0)
        self.vocab = max_word_in_sparse_text(self.comments, self.vocab)
        self.vocab = max_word_in_sparse_text(self.labeled, self.vocab)
        self.vocab = max_word_in_sparse_text(self.background, self.vocab)

        self.W = self.vocab + 1
        self.D = len(self.documents)
        self.L = len(self.labeled)
        self.B = len(self.background)

        self.optimize_documents()

    def iterdocuments(self):
        """Documents are computed in E-step in turn. 
            Yields generator of documents.  
            In this case, 2-tuples of (document,comment).
        """
        return izip(self.documents, self.comments)

    def iterlabeled(self):
        """Documents are computed in E-step in turn. 
            Yields generator of documents.  
            In this case, 2-tuples of (document,y).
        """
        return izip(self.labeled, self.y)

    def iterbackground(self):
        """Documents are computed in E-step in turn. 
            Yields generator of documents.  
        """
        return self.background

    def optimize_documents(self):
        """Converts the local documents from sparse representation into normal vector."""
        # OPTIMIZATION: turn all documents into arrays
        self.documents = [topiclib.doc_to_array(d) for d in self.documents]
        self.comments = [topiclib.doc_to_array(d) for d in self.comments]
        self.labeled = [topiclib.doc_to_array(d) for d in self.labeled]
        self.background = [topiclib.doc_to_array(d) for d in self.background]

    def initialize(self, Ku, Ks, Kb):
        """Accepts K number of topics in document.
            Initializes all of the hidden variable arrays now that it knows dimensions
            of topics, vocabulary, etc.
        """
        assert self.documents is not None
        assert Ku is not None
        assert Ks is not None
        assert Kb is not None

        K = Ku + Ks + Kb

        # give at least more documents than topics
        # so that it's not singular
        assert self.D > K

        self.K = K
        self.Ku = Ku
        self.Ks = Ks
        self.Kb = Kb

        self.Kc = self.Ku + self.Ks
        self.Kl = self.Ks + self.Kb

        W = self.W

        # Initialize the variational distribution q(beta|lambda)
        self.beta = topiclib.initialize_beta(K, W)

        # "it suffices to fix alpha to uniform 1/K"
        # initialize to ones so that the topics are more evenly distributed
        # good for small datasets
        self.alphaU = np.ones((Ku,)) * (1.0 / Ku)
        self.alphaS = np.ones((Ks,)) * (1.0 / Ks)
        self.alphaB = np.ones((Kb,)) * (1.0 / Kb)

        # todo: not using this yet
        #self.alphaD = ...
        
        def uniform_phi(Nds, size):
            D = len(Nds)
            return [(np.ones((Nds[d], size)) * (1.0 / size)) for d in xrange(D)]

        document_Nds = self.num_words_per(self.documents)
        self.phiD = uniform_phi(document_Nds, self.Ku)
        comment_Nds = self.num_words_per(self.comments)
        self.phiC = uniform_phi(comment_Nds, self.Kc)
        labeled_Nds = self.num_words_per(self.labeled)
        self.phiL = uniform_phi(labeled_Nds, self.Kl)
        background_Nds = self.num_words_per(self.background)
        self.phiB = uniform_phi(background_Nds, self.Kb)

        self.gammaD = np.ones((self.D, self.Ku)) * (1.0 / self.Ku)
        self.gammaC = np.ones((self.D, self.Kc)) * (1.0 / self.Kc)
        self.gammaL = np.ones((self.L, self.Kl)) * (1.0 / self.Kl)
        self.gammaB = np.ones((self.B, self.Kb)) * (1.0 / self.Kb)
        graphlib.initialize_random(self.gammaD)
        graphlib.initialize_random(self.gammaC)
        graphlib.initialize_random(self.gammaL)
        graphlib.initialize_random(self.gammaB)

        self.eta = graphlib.random_normal(0, 2.0, (Ks,))
        self.sigma_squared = 0.5

        print 'eta start: {0}'.format(self.eta)

        self.is_initialized = True

    def to_dict(self):
        return { 
    'eta': self.eta, 'sigma_squared': self.sigma_squared,
    'beta': self.beta, 
    'gammaD': self.gammaD, 'gammaC': self.gammaC, 'gammaL': self.gammaL, 'gammaB': self.gammaB,
    'phiD': self.phiD, 'phiC': self.phiC, 'phiL': self.phiL, 'phiB': self.phiB,
    }

def tlc_e_step(global_iterations, v):
    total_local_i = 0
    dlocal_i, clocal_i, local_i, blocal_i  = 0,0,0,0
    for d, (document, comment) in enumerate(v.iterdocuments()):
        # todo: this should be more complicated in order to share strength
        dlocal_i = topiclib.lda_E_step_for_doc(global_iterations, 
                                                dlocal_i,
                                                d, document,
                                                v.alphaU, v.beta[:v.Ku],
                                                v.gammaD[d], v.phiD[d])

        alphaC = np.concatenate((v.alphaU, v.alphaS))
        clocal_i = topiclib.lda_E_step_for_doc(global_iterations,
                                                clocal_i,
                                                d, comment,
                                                alphaC, v.beta[:v.Kc],
                                                v.gammaC[d], v.phiC[d])
    total_local_i += dlocal_i
    total_local_i += clocal_i
        

    for l, (labeled_doc,y) in enumerate(v.iterlabeled()):
        alphaL = np.concatenate((v.alphaS, v.alphaB))
        local_i = topiclib.partial_slda_E_step_for_doc(global_iterations,
                                                        local_i,
                                                        l, labeled_doc, y,
                                                        alphaL, v.beta[-v.Kl:],
                                                        v.gammaL[l], v.phiL[l],
                                                        v.eta, v.sigma_squared)
    total_local_i += local_i

    for b, bg_doc in enumerate(v.iterbackground()):
        blocal_i = topiclib.lda_E_step_for_doc(global_iterations, 
                                                blocal_i,
                                                b, bg_doc,
                                                v.alphaB, v.beta[-v.Kb:],
                                                v.gammaB[b], v.phiB[b])
    total_local_i += blocal_i

    return total_local_i

def tlc_m_step(var):
    ### M-step: ###
    print 'updating betas..'
    Ku, Ks, Kb = var.Ku, var.Ks, var.Kb
    
    # update unlabeled document topics
    dc = var.documents + var.comments
    phi_dc = var.phiD + [p[:,:Ku] for p in var.phiC]
    topiclib.lda_recalculate_beta(dc, var.beta[:Ku], phi_dc)

    # update sentiment topics
    cl = var.comments + var.labeled
    phi_cl = [p[:,-Ks:] for p in var.phiC] + [p[:,:Ks] for p in var.phiL]
    topiclib.lda_recalculate_beta(cl, var.beta[Ku:Ku+Ks], phi_cl)

    # update background topics
    lb = var.labeled + var.background
    phi_lb = [p[:,-Kb:] for p in var.phiL] + var.phiB
    topiclib.lda_recalculate_beta(lb, var.beta[-Kb:], phi_lb)

    print 'eta sigma...'
    # update response variable gaussian global parameters
    var.sigma_squared = topiclib.partial_slda_recalculate_eta_sigma(var.eta, var.y, var.phiL)


def tlc_print_func(i, var):
    #print 'y: %s' % var.y
    print 'eta: %s' % var.eta
    print 'ss: %s' % var.sigma_squared
    
    if i % 5 == 0:
        jsondata.save('mytlc-output-%s.dat' % i, var.to_dict())
    # todo: predict y and calculate error

def tlc_global_elbo(v):
    # use equivalent of pSLDA global elbo just for this since it's the important part
    # it's more efficient to calculate than true elbo
    return np.sum(topiclib.partial_slda_local_elbo(v.labeled[d], v.y[d], v.alphaL, v.beta[-v.Kl:], v.gammaL[d], v.phiL[d], v.eta, v.sigma_squared) for d in xrange(len(v.labeled)))

run_tlc = partial(graphlib.run_variational_em, 
                    e_step_func=tlc_e_step, 
                    m_step_func=tlc_m_step, 
                    global_elbo_func=tlc_global_elbo,
                    print_func=tlc_print_func)


            
if __name__=='__main__':
    dirname = 'synthtlc'
    dirname = 'synthbig'

    # use my tlc synthetically generated dataset
    documents = topiclib.read_sparse(dirname + '/documents.dat')
    comments = topiclib.read_sparse(dirname + '/comments.dat')
    labeled_documents = topiclib.read_sparse(dirname + '/labeled.dat')
    background = topiclib.read_sparse(dirname + '/background.dat')

    y = np.loadtxt(dirname + '/yL.npy')
    real_data = (documents, comments, labeled_documents, background, y)

    var = TLCVars(real_data, Ku=29, Ks=5, Kb=24)

    try:
        output = run_tlc(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

