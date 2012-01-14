#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, time, string
import numpy as ny
from scipy.special import psi


def dirichlet_expectation(alpha):
    """
    From Matt Hoffman:
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def row_normalize(matrix):
    """Accepts 2-D matrix.
        
    """
    (nrows, ncols) = matrix.shape
    rowsums = ny.sum(matrix, axis=1)
    matrix.divide(matrix, rowsums.reshape(nrows, 1), matrix)
    return matrix

def initialize_beta(num_topics, num_words):
    """Initializes beta randomly using a random dirichlet.
        Accepts integers number of topics, and number of words in vocab.
        Returns a TxW matrix which have the probabilities of word
            distributions.  Each row sums to 1.
    """
    l = 1*ny.random.gamma(100., 1./100., (num_topics, num_words))
    Elogbeta = dirichlet_expectation(l)
    expElogbeta = ny.exp(Elogbeta)
    
    # todo: jperla: do I need to normalize? 
    # otherwise word prob doesn't sum to 1?!

    return row_normalize(expElogbeta)

def initialize_uniform(matrix):
    """Accepts a matrix with a defined shape.
        Initializes it to to be uniform probability on row.
        Each row on last dimension should sum to 1.
        Returns the original matrix, modified.
    """
    (nrows, ncols) = matrix.shape
    matrix = numpy.ones(matrix.shape)*(1/ncols)
    return matrix

def elbo_did_not_converge(elbo, last_elbo):
    """Takes two elbo doubles.  
        Returns boolean.
        Figures out whether the elbo is sufficiently smaller than
            last_elbo.
    """
    print 'elbo is {0}'.format(elbo)
    if elbo == 0 or last_elbo == 0:
        return True
    else:
        # todo: do a criterion convergence test
        return True


def run_em(documents):
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

    """
    K = 3
    J = 3
    W = 10
    elbo = 0
    last_elbo = 0


    # "it suffices to fix alpha to uniform 1/K"
    alpha = initialize_uniform(numpy.array((1, K)))

    D = len(documents)

    # Initialize the variational distribution q(beta|lambda)
    betaD = initialize_beta(K, W)
    betaC = initialize_beta(J, W)

    phiD = numpy.zeros((D, W, K))
    gammaD = numpy.zeros((D, K))

    phiC = numpy.zeros((D, W, K))
    gammaC = numpy.zeros((D, K))

    y = numpy.ones((D,))

    eta = numpy.ones((D,))
    sigma_squared = 0

    iterations = 0
    while elbo_did_not_converge(elbo, last_elbo):
        for d, (document, comment) in enumerate(documents):
            # do E-step on document first
            do_E_step(phiD[d], gammaD[d], document)
            # do E-step on comment next
            do_E_step(phiC[d], gammaC[d], comment)

            # update the response variable
              # todo: this is wrong, methinks!
            y[d] = eta.T * ny.sum(phiD[d], axis=0) / N
        # do the M-step:
        # update topics: βk,wnew ∝ ΣdΣn 1(wd,n = w) φkd,n
         # todo: haven't done this

        # update response variable gaussian global parameters:
        # ηnew ← (E[ATA])-1 E[A]Ty
        # σ2new ← (1/D) {yTy - yTE[A]ηnew}
          # todo: calculate

        last_elbo = local_elbo
        elbo = calculate_elbo()

        # maybe write all these vars every iteration (or every 10)

        iterations += 1
    return (iterations, elbo, y, (betaD, betaC), (phiD, phiC), (gammaD, gammaC))

def do_E_step(phi, gamma, doc):
    """Given phi and gamma matrices and document of the document.
        Recalculate phi and gamma repeatedly iteratively.
        Uses local elbo calculation to check for convergence.
    """
    initialize_uniform(phi)
    local_elbo, local_last_elbo = 0, 0
    print "starting E step"
    i = 0
    while elbo_did_not_converge(local_elbo, local_last_elbo):
        # update gamma: γnew ← α + Σnφn
        gamma = alpha + ny.sum(phi, axis=0)
        # update phid:
        # φjnew ∝ exp{ E[log θ|γ] + 
        #              E[log p(wj|β1:K)] + 
        #              (y / Nσ2) η  — 
        #              [2(ηTφ-j)η + (η∘η)] / (2N2σ2) }

        # todo: update phi; speed this up

        # calculate new ELBO
        local_last_elbo = local_elbo
        local_elbo = calculate_elbo()
        i += 1
        print "e-step iteration {0}".format(i)
    print "done e-step: {0} iterations".format(i)

def calculate_elbo():
    """Given all of the parametes.
        Calculate the evidence lower bound.
        Helps you know when convergence happens.
    """
    return 0.0

            
if __name__=='__main__':
    # todo: these should probably be input
    # documents are 2-tuples of document, comment
    test_data = [(1, 2), (3, 4),]

    output = run_em(test_data)

