#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
    a library for working with topic graphical models like LDA and sLDA

    Also works with partial sLDA and TLC. Uses graphlib.
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""

from itertools import repeat

try:
    import numpypy as np
except ImportError:
    import numpy as np
    np.seterr(invalid='raise')

try:
    from scipy.special import gammaln
except ImportError:
    from scipypy import gammaln

import graphlib
from graphlib import logsumexp
from graphlib import logdotexp
from graphlib import ensure

def initialize_log_beta(num_topics, num_words):
    """Initializes beta randomly using a random dirichlet.
        Accepts integers number of topics, and number of words in vocab.
        Returns a TxW matrix which have the probabilities of word
            distributions.  Exp of each row sums to 1.
    """
    l = 1*np.random.gamma(100., 1./100., (num_topics, num_words))
    Elogbeta = graphlib.dirichlet_expectation(l)
    beta = graphlib.log_row_normalize(Elogbeta)
    return beta

def initialize_beta(num_topics, num_words):
    """Initializes beta randomly using a random dirichlet.
        Accepts integers number of topics, and number of words in vocab.
        Returns a TxW matrix which have the probabilities of word
            distributions.  Each row sums to 1.
    """
    log_beta = initialize_log_beta(num_topics, num_words)
    return np.exp(log_beta)

def lda_recalculate_beta(text, beta, phi):
    """
    update topics: βk,wnew ∝ ΣdΣn 1(wd,n = w) φkd,n

    Accepts beta matrix (KxW) and 
        phi, a D-length list of (N x K) matrices.
    """
    (K,W) = beta.shape
    D = len(phi)
    beta[:,:] = np.zeros(beta.shape)    

    if isinstance(text[0], np.ndarray):
        for d in xrange(D):
            ensure(phi[d].shape[1] == K)
            for n,word in enumerate(text[d]):
                beta[:,word] += phi[d][n,:]
            #words, indexes = text[d], np.array(range(len(text[d])))
            #beta[:,words] += phi[d][indexes,:].T
    else:
        for d in xrange(D):
            ensure(phi[d].shape[1] == K)
            for n,word,count in iterwords(text[d]):
                beta[:,word] += phi[d][n,:]
    graphlib.row_normalize(beta)
    return beta

def lda_recalculate_log_beta(text, log_beta, log_phi):
    """
    update topics: βk,wnew ∝ ΣdΣn 1(wd,n = w) φkd,n

    Accepts log beta matrix (KxW) and 
        log phi, a D-length list of (N x K) matrices.
    """
    (K,W) = log_beta.shape
    D = len(log_phi)

    # todo: jperla: should use -inf or a different really small number?!
    log_beta[:,:] = np.ones(log_beta.shape) * float('-1000')
    
    if isinstance(text[0], np.ndarray):
        for d in xrange(D):
            for n,word in enumerate(text[d]):
                for k in xrange(K):
                    log_beta[k,word] = np.logaddexp(log_beta[k,word], log_phi[d][n][k])
    else:
        for d in xrange(D):
            for n,word,count in iterwords(text[d]):
                for k in xrange(K):
                    log_beta[k,word] = np.logaddexp(log_beta[k,word], log_phi[d][n][k])
    graphlib.log_row_normalize(log_beta)
    return log_beta


def calculate_big_phi(phi1, phi2):
    """ Pretends that two separate sets of D phi matrices (Nd x K) each
        are one big phi matrix.
        This is needed for the latent model.
        The trick is to make a big matrix with four quadrants.
        The top left quadrant has the phi1 matrix, the bottom right has phi2.
        The remaining two quadrants are filled with zeros.
    """
    (n1, k1) = phi1.shape
    (n2, k2) = phi2.shape
    big_phi = np.zeros((n1 + n2, k1 + k2))
    big_phi[0:n1,0:k1] = phi1
    big_phi[n1:n1+n2,k1:k1+k2] = phi2
    return big_phi

def calculate_big_log_phi(phi1, phi2):
    """ Pretends that two separate sets of D phi matrices (Nd x K) each
        are one big phi matrix.
        This is needed for the latent model.
        The trick is to make a big matrix with four quadrants.
        The top left quadrant has the phi1 matrix, the bottom right has phi2.
        The remaining two quadrants are filled with zeros.
    """
    (n1, k1) = phi1.shape
    (n2, k2) = phi2.shape
    big_phi = np.ones((n1 + n2, k1 + k2)) * float('-1000')
    big_phi[0:n1,0:k1] = phi1
    big_phi[n1:n1+n2,k1:k1+k2] = phi2
    return big_phi

def slda_recalculate_eta_sigma(eta, y, phi):
    """
        Accepts eta (K)-size vector,
            also y (a D-size vector of reals),
            also a phi D-size vectors of NxK matrices.
        Returns new sigma squared update (a double).

        ηnew ← (E[ATA])-1 E[A]Ty
        σ2new ← (1/D) {yTy - yTE[A]ηnew}

        (Note that A is the D X (K + J) matrix whose rows are the vectors ZdT.)
        (Also note that the dth row of E[A] is φd, and E[ATA] = Σd E[ZdZdT] .)
        (Also, note that E[Z] = φ := (1/N)Σnφn, and E[ZdZdT] = (1/N2)(ΣnΣm!=nφd,nφd,mT  +  Σndiag{φd,n})

    """
    ensure(len(eta) == phi.shape[1])
    return partial_slda_recalculate_eta_sigma(eta, y, phi)

def partial_slda_recalculate_eta_sigma(eta, y, phi):
    """
        Same as slda_recalculate_eta_sigma, but also
          supports partial updates if len(eta) < phi.shape[1] .
          Will only update based on first Ks topics of phi
    """
    D = len(phi)
    ensure(D >= 1)

    N,K = phi[0].shape
    Ks = len(eta)

    print 'e_a...'
    E_A = np.empty((D, Ks))
    for d in xrange(D):
        E_A[d,:] = calculate_EZ(phi[d][:,:Ks])
  
    E_ATA_inverse = calculate_E_ATA_inverse([p[:,:Ks] for p in phi])

    print 'new eta...'
    new_eta = np.dot(np.dot(E_ATA_inverse, E_A.T), y)
    eta[:] = new_eta
    
    print 'new sigma squared...'
    new_sigma_squared = (1.0 / D) * (np.dot(y, y) - np.dot(np.dot(y, E_A), eta))
    return new_sigma_squared

def lm_recalculate_eta_sigma(eta, y, phi1, phi2):
    """
        Accepts eta (K+J)-size vector,
            also y (a D-size vector of reals),
            also two phi D-size vectors of NxK matrices.
        Returns new sigma squared update (a double).

        ηnew ← (E[ATA])-1 E[A]Ty
        σ2new ← (1/D) {yTy - yTE[A]ηnew}

        (Note that A is the D X (K + J) matrix whose rows are the vectors ZdT for document and comment concatenated.)
        (Also note that the dth row of E[A] is φd, and E[ATA] = Σd E[ZdZdT] .)
        (Also, note that E[Z] = φ := (1/N)Σnφn, and E[ZdZdT] = (1/N2)(ΣnΣm!=nφd,nφd,mT  +  Σndiag{φd,n})
    """
    ensure(len(phi1) == len(phi2))
    D = len(phi1)

    Nd,K = phi1[0].shape
    Nc,J = phi2[0].shape
    Ndc, KJ = (Nd+Nc,K+J)

    #print 'e_a...'
    E_A = np.zeros((D, KJ))
    for d in xrange(D):
        E_A[d,:] = calculate_EZ_from_small_phis(phi1[d], phi2[d])
  
    #print 'inverse...'
    E_ATA_inverse = calculate_E_ATA_inverse_from_small_phis(phi1, phi2)

    #print 'new eta...'
    #new_eta = matrix_multiply(matrix_multiply(E_ATA_inverse, E_A.T), y)
    new_eta = np.dot(np.dot(E_ATA_inverse, E_A.T), y)
    if np.sum(np.abs(new_eta)) > (KJ * KJ * 5):
        print 'ETA is GOING CRAZY {0}'.format(eta)
        print 'aborting the update!!!'
    else:
        eta[:] = new_eta
    
    # todo: don't do this later
    # keep sigma squared fix
    #import pdb; pdb.set_trace()
    #new_sigma_squared = (1.0 / D) * (np.dot(y, y) - np.dot(np.dot(np.dot(np.dot(y, E_A), E_ATA_inverse), E_A.T), y))
    new_sigma_squared = 1.0
    return new_sigma_squared


def calculate_EZ_from_small_phis(phi1, phi2):
    """
        Accepts a two small phi matrices (like (NdxK) and (NcxJ))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc = phi1.shape[0] + phi2.shape[0]
    ez = np.concatenate((np.sum(phi1, axis=0), np.sum(phi2, axis=0)), axis=1)
    return ez / Ndc

def calculate_EZ_from_small_log_phis(log_phi1, log_phi2):
    """
        Accepts a two small phi matrices (like (NdxK) and (NcxJ))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc = log_phi1.shape[0] + log_phi2.shape[0]
    ez = np.concatenate((logsumexp(log_phi1, axis=0), logsumexp(log_phi2, axis=0)), axis=1)
    return ez - np.log(Ndc)
    
def calculate_EZ(big_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    N,K = big_phi.shape
    return np.sum(big_phi, axis=0) / N

    
def calculate_EZ_from_big_log_phi(big_log_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc,KJ = big_log_phi.shape
    return logsumexp(big_log_phi, axis=0) - np.log(Ndc)

def calculate_EZZT_from_small_phis(phi1, phi2):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[ZdZdT].
        Returns the final matrix ((K+J) x (K+J)).

        (Also, E[ZdZdT] = (1/N2)(ΣNΣm!=nφd,nφd,mT  +  ΣNdiag{φd,n})
    """
    Nd,K = phi1.shape
    Nc,J = phi2.shape
    (Ndc, KJ) = (Nd+Nc, K+J)
    inner_sum = np.zeros((KJ, KJ))

    p1 = np.matrix(phi1)
    p2 = np.matrix(phi2)

    for i in xrange(K):
        for j in xrange(K):
            m = np.dot(np.matrix(p1[:,i]), np.matrix(p1[:,j]).T)
            inner_sum[i,j] = np.sum(m) - np.sum(np.diagonal(m))

    for i in xrange(J):
        for j in xrange(J):
            m = np.dot(np.matrix(p2[:,i]), np.matrix(p2[:,j]).T)
            inner_sum[K+i,K+j] = np.sum(m) - np.sum(np.diagonal(m))

    for i in xrange(K):
        for j in xrange(J):
            m = np.dot(np.matrix(p1[:,i]), np.matrix(p2[:,j]).T)
            inner_sum[i,K+j] = np.sum(m)

    for i in xrange(J):
        for j in xrange(K):
            m = np.dot(np.matrix(p2[:,i]), np.matrix(p1[:,j]).T)
            inner_sum[K+i,j] = np.sum(m)

    big_phi_sum = np.concatenate((np.sum(phi1, axis=0),
                                  np.sum(phi2, axis=0)), axis=1)
    ensure(big_phi_sum.shape == (KJ,))
    inner_sum += np.diagonal(big_phi_sum)

    inner_sum /= (Ndc * Ndc)
    return inner_sum

def calculate_EZZT_from_small_log_phis(phi1, phi2):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[ZdZdT].
        Returns the final matrix ((K+J) x (K+J)).

        (Also, E[ZdZdT] = (1/N2)(ΣNΣm!=nφd,nφd,mT  +  ΣNdiag{φd,n})
    """
    Nd,K = phi1.shape
    Nc,J = phi2.shape
    (Ndc, KJ) = (Nd+Nc, K+J)
    inner_sum = np.zeros((KJ, KJ))

    p1 = np.matrix(phi1)
    p2 = np.matrix(phi2)

    for i in xrange(K):
        for j in xrange(K):
            m = logdotexp(np.matrix(p1[:,i]), np.matrix(p1[:,j]).T)
            m += np.diagonal(np.ones(Nd) * -1000)
            inner_sum[i,j] = logsumexp(m.flatten())

    for i in xrange(J):
        for j in xrange(J):
            m = logdotexp(np.matrix(p2[:,i]), np.matrix(p2[:,j]).T)
            m += np.diagonal(np.ones(Nc) * -1000)
            inner_sum[K+i,K+j] = logsumexp(m.flatten())

    for i in xrange(K):
        for j in xrange(J):
            m = logdotexp(np.matrix(p1[:,i]), np.matrix(p2[:,j]).T)
            inner_sum[i,K+j] = logsumexp(m.flatten())

    for i in xrange(J):
        for j in xrange(K):
            m = logdotexp(np.matrix(p2[:,i]), np.matrix(p1[:,j]).T)
            inner_sum[K+i,j] = logsumexp(m.flatten())

    big_phi_sum = np.concatenate((logsumexp(phi1, axis=0),
                                  logsumexp(phi2, axis=0)), axis=1)
    ensure(big_phi_sum.shape == (KJ,))
    for i in xrange(KJ):
        inner_sum[i,i] = logsumexp([inner_sum[i,i], big_phi_sum[i]])

    inner_sum -= np.log(Ndc * Ndc)
    return inner_sum

def calculate_EZZT(big_phi):
    """
        Accepts a big phi matrix (like (N x K)
        Calculates E[ZdZdT].
        Returns the final matrix (K x K).

        (Also, E[ZdZdT] = (1/N2)(ΣNΣm!=nφd,nφd,mT  +  ΣNdiag{φd,n})
    """
    (N, K) = big_phi.shape
    inner_sum = np.empty((K, K))

    for i in xrange(K):
        for j in xrange(K):
            inner_sum[i,j] = np.sum(np.multiply.outer(big_phi[:,i], big_phi[:,j])) - np.sum(np.dot(big_phi[:,i], big_phi[:,j]))
    inner_sum += np.diag(np.sum(big_phi, axis=0))
    inner_sum /= (N * N)
    return inner_sum

def calculate_E_ATA_inverse(phi):
    """Accepts number of documents, 
        and a big_phi matrix of size (N, K).
        Returns a new matrix which is inverse of E([ATA]) of size (K,K).
    """
    print 'E_ATA...'
    D = len(phi)
    N,K = phi[0].shape
    E_ATA = sum(calculate_EZZT(phi[d]) for d in xrange(D))
    ensure(E_ATA.shape == (K, K))

    print 'inverse...'
    # todo: does not work in pypy
    return np.linalg.inv(E_ATA)

def calculate_E_ATA_inverse_from_small_phis(phi1, phi2):
    """Accepts number of documents, 
        and two small phi matrices of size (Nd,K) and (Nc,J).
        Returns a new matrix which is inverse of E([ATA]) of size (K+J,K+J).

        (Note that A is the D X (K + J) matrix whose rows are the vectors ZdT for document and comment concatenated.)
    """
    D = len(phi1)
    Nd,K = phi1[0].shape
    Nc,J = phi2[0].shape
    (Ndc, KJ) = (Nd+Nc, K+J)
    E_ATA = sum(calculate_EZZT_from_small_phis(phi1[d], phi2[d]) for d in xrange(D))
    ensure(E_ATA.shape == (KJ, KJ))

    # todo: this does not work in pypy
    return np.linalg.inv(E_ATA)

def lda_update_gamma(alpha, phi, gamma):
    """
     Accepts:
        gamma and alpha are K-size vectors.
        Phi is an NxK vector.
     Returns gamma.

     update gamma: γnew ← α + Σnφn
    """
    ensure(phi.shape[1] == len(gamma))
    gamma[:] = alpha + np.sum(phi, axis=0)
    return gamma


def lda_update_log_gamma(log_alpha, log_phi, log_gamma):
    """
     Same as lda_update_gamma, 
        but in log probability space.
    """
    ensure(log_phi.shape[1] == len(log_gamma))
    log_gamma[:] = logsumexp([log_alpha, logsumexp(log_phi, axis=0)], axis=0)
    return log_gamma


def _unoptimized_slda_update_phi(text, phi, gamma, beta, y_d, eta, sigma_squared):
    """
        Update phi in LDA. 
        phi is N x K matrix.
        gamma is a K-size vector

     update phid:
     φd,n ∝ exp{ E[log θ|γ] + 
                 E[log p(wn|β1:K)] + 
                 (y / Nσ2) η  — 
                 [2(ηTφd,-n)η + (η∘η)] / (2N2σ2) }
     
     Note that E[log p(wn|β1:K)] = log βTwn
    """
    (N, K) = phi.shape
    #assert len(eta) == K
    #assert len(gamma) == K
    #assert beta.shape[0] == K

    phi_sum = np.sum(phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = graphlib.dirichlet_expectation(gamma)
    ensure(len(ElogTheta) == K)

    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = (eta * eta)
    front = (-1.0 / (2 * N * Ns))

    for n,word,count in iterwords(text):
        phi_sum -= phi[n]
        ensure(len(phi_sum) == K)

        pB = np.log(beta[:,word])
        pD = (front * (((2 * np.dot(eta, phi_sum) * eta) + eta_dot_eta))
                            )
        ensure(len(pB) == K)
        ensure(len(pC) == K)
        ensure(len(pD) == K)

        # must exponentiate and sum immediately!
        #phi[n,:] = np.exp(ElogTheta + pB + pC + pD)
        #phi[n,:] /= np.sum(phi[n,:])
        # log normalize before exp for numerical stability
        phi[n,:] = ElogTheta + pB + pC + pD
        phi[n,:] -= graphlib.logsumexp(phi[n,:])
        phi[n,:] = np.exp(phi[n,:])

        # add this back into the sum
        # unlike in LDA, this cannot be computed in parallel
        phi_sum += phi[n]

    return phi

def doc_to_array(doc):
    """Accepts a list of 2-tuples, first being integer word id, second word count.
        Returns a numpy array representing the full document.
        An array of wordids with the length being the sum of the word counts.
    """
    return np.array([w for word,count in doc for w in repeat(word, count)])

def lda_update_phi(text, phi, gamma, beta, normalize=True, logspace=False):
    """
        Update phi in LDA. 
        phi is N x K matrix.
        gamma is a K-size vector

     update phid:
     φd,n ∝ exp{ E[log θ|γ] + 
                 E[log p(wn|β1:K)] }
     
     Note that E[log p(wn|β1:K)] = log βTwn
    """
    (N, K) = phi.shape

    ElogTheta = graphlib.dirichlet_expectation(gamma)

    # todo: call a log version of this in slda and others!
    ensure(isinstance(text, np.ndarray))
    phi[:,:] = ElogTheta + np.log(beta[:,text].T)
    if normalize:
        graphlib.log_row_normalize(phi)
    if not logspace:
        phi[:,:] = np.exp(phi[:,:])
    return phi

def slda_update_phi(text, phi, gamma, beta, y_d, eta, sigma_squared):
    """Update phi in (supervised!) sLDA. 
       phi is N x K matrix.
       gamma is a K-size vector

    update phid:
    φd,n ∝ exp{ E[log θ|γ] + 
                E[log p(wn|β1:K)] + 
                (y / Nσ2) η  — 
                [2(ηTφd,-n)η + (η∘η)] / (2N2σ2) }
           exp{ A + B + C - D}
    
    Note that E[log p(wn|β1:K)] = log βTwn

    If len(eta) < phi.shape[1], then it is a Partial update.
        Same as slda update phi, but eta only acts on first few topics in phi.
    """
    ensure(len(eta) == phi.shape[1])
    partial_slda_update_phi(text, phi, gamma, beta, y_d, eta, sigma_squared)

def partial_slda_update_phi(text, phi, gamma, beta, y_d, eta, sigma_squared):
    """Same as slda update phi, but eta may be smaller than total number of topics.
        So only some of the topics contribute to y.
    """
    (N, K) = phi.shape
    Ks = len(eta)

    phi_sum = np.sum(phi[:,:Ks], axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = graphlib.dirichlet_expectation(gamma)

    front = (-1.0 / (2 * N * Ns))
    eta_dot_eta = front * (eta * eta)
    pC = ((1.0 * y_d / Ns) * eta) + eta_dot_eta

    right_eta_times_const = (front * 2 * eta)

    if isinstance(text, np.ndarray):
        # if text is in array form, do an approximate fast matrix update
        phi_minus_n = -(phi[:,:Ks] - phi_sum)
        phi[:,:] = ElogTheta + np.log(beta[:,text].T)
        phi[:,:Ks] += pC
        phi[:,:Ks] += np.dot(np.matrix(np.dot(phi_minus_n, eta)).T, np.matrix(right_eta_times_const))
        graphlib.log_row_normalize(phi)
        phi[:,:] = np.exp(phi[:,:])
    else:
        # otherwise, iterate through each word
        for n,word,count in iterwords(text):
            phi_sum -= phi[n,:Ks]

            pB = np.log(beta[:,word])
            pD = (np.dot(eta, phi_sum) * right_eta_times_const) 

            # must exponentiate and normalize immediately!
            phi[n,:] = ElogTheta + pB
            phi[n,:] += pC + pD
            phi[n,:] -= graphlib.logsumexp(phi[n,:]) # normalize in logspace
            phi[n,:] = np.exp(phi[n,:])


            # add this back into the sum
            # unlike in LDA, this cannot be computed in parallel
            phi_sum += phi[n,:Ks]
    return phi


def slda_update_log_phi(text, log_phi, log_gamma, log_beta, y_d, eta, sigma_squared):
    """
        Same as update_phi_lda_E_step but in log probability space.
    """
    (N, K) = log_phi.shape

    log_phi_sum = logsumexp(log_phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = graphlib.dirichlet_expectation(np.exp(log_gamma))

    front = (-1.0 / (2 * N * Ns))
    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = front * (eta * eta)
    log_const = np.log(ElogTheta + pC + eta_dot_eta)

    log_right_eta_times_const = np.log(front * 2 * eta)

    ensure(isinstance(text, np.ndarray))

    # if text is in array form, do an approximate fast matrix update
    log_phi_minus_n = -1 + (logsumexp([log_phi, (-1 + log_phi_sum)]))

    log_phi[:,:] = logsumexp([log_beta[:,text].T, 
                              logdotexp(np.matrix(logdotexp(log_phi_minus_n, np.log(eta))).T, 
                                        np.matrix(log_right_eta_times_const)), 
                              log_const,], axis=0)

    graphlib.log_row_normalize(log_phi)

    return log_phi

def partial_slda_global_elbo(v):
    return np.sum(partial_slda_local_elbo(v.documents[d], v.y[d], v.alpha, v.beta, v.gamma[d], v.phi[d], v.eta, v.sigma_squared) for d in xrange(len(v.documents)))

def partial_slda_local_elbo(document, y, alpha, beta, gamma, phi, eta, sigma_squared):
    """Same as slda local elbo, but different elbo y update.
        Only send part of phi!
    """
    elbo = 0.0
    #print 'elbo lda terms...'
    elbo += lda_elbo_terms(document, alpha, beta, gamma, phi)

    #print 'elbo slda y...'
    Ks = len(eta)
    elbo += slda_elbo_y(y, eta, phi[:,:Ks], sigma_squared)

    #print 'elbo entropy...'
      # todo: is it just the same as lda??
    elbo += lda_elbo_entropy(gamma, phi)
    return elbo

def slda_global_elbo(v):
    return np.sum(slda_local_elbo(v.documents[d], v.y[d], v.alpha, v.beta, v.gamma[d], v.phi[d], v.eta, v.sigma_squared) for d in xrange(len(v.documents)))

def slda_local_elbo(document, y, alpha, beta, gamma, phi, eta, sigma_squared):
    """Given all of the parametes for one document.
        Calculate the evidence lower bound.
        Helps you know when convergence happens in E step.

   ELBO = ℒ = 
    E[log p(θ|α)] + ΣNE[log p(Zn|θ)] + ΣNE[log p(wn|Zn,β1:K)]
    E[log p(y|Z1:N,η,σ2)] + H(q)

    The first 3 terms are LDA terms 
        for document:

    The last terms are for the y, and for the entropy of q.
    """
    elbo = 0.0
    #print 'elbo lda terms...'
    elbo += lda_elbo_terms(document, alpha, beta, gamma, phi)

    #print 'elbo slda y...'
    elbo += slda_elbo_y(y, eta, phi, sigma_squared)

    #print 'elbo entropy...'
      # todo: is it just the same as lda??
    elbo += lda_elbo_entropy(gamma, phi)
    return elbo


def lda_global_elbo(v):
    return np.sum(lda_local_elbo(v.documents[d], v.alpha, v.beta, v.gamma[d], v.phi[d],) for d in xrange(len(v.documents)))

def lda_local_elbo(document, alpha, beta, gamma, phi):
    return lda_elbo_terms(document, alpha, beta, gamma, phi) + lda_elbo_entropy(gamma, phi)

def lda_E_step_for_doc(global_iteration, 
                        last_local_iterations,
                        d, document,
                        alpha, beta,
                        gamma, phi):
    """Given phi and gamma matrices and document of the document.
        Recalculate phi and gamma repeatedly iteratively.
        Uses local elbo calculation to check for convergence.
    """
    #print "starting E step on doc {0}".format(d)
    graphlib.initialize_random(phi)

    ensure(phi.shape[1] == beta.shape[0] == len(gamma) == len(alpha))
    ensure(phi.shape[0] == len(document))

    i = 0
    min_iter = 20 - global_iteration
    max_iter = last_local_iterations if last_local_iterations > 0 else 20
    last_local_elbo, local_elbo = graphlib.INITIAL_ELBO - 100, graphlib.INITIAL_ELBO
    while graphlib.elbo_did_not_converge(local_elbo, last_local_elbo, i, 
                                        criterion=0.1, 
                                        min_iter=min_iter, max_iter=max_iter):
        #print 'will update gamma...'
        lda_update_gamma(alpha, phi, gamma)

        #print 'will update phis...'
        lda_update_phi(document, phi, gamma, beta)

        if last_local_iterations == 0:
            #print 'will calculate elbo...'
            last_local_elbo = local_elbo
            local_elbo = lda_local_elbo(document, alpha, beta, gamma, phi)
        i += 1

        #print {'beta': beta, 'gamma': gamma, 'phi': phi}
        #print "{2}: e-step iteration {0} ELBO: {1}".format(i, local_elbo, global_iteration)
    if d % 100 == 0:
        print "{2}: done LDA e-step on doc {3}: {0} iterations ELBO: {1}".format(i, local_elbo, global_iteration, d)
    return i

def partial_slda_E_step_for_doc(global_iteration, 
                                last_local_iterations,
                                d, document, y,
                                alpha, beta, gamma, phi,
                                eta, sigma_squared):
    """Same as sLDA e-step, but slightly different phi update,
        and slightly different elbo calculation.
    """
    #print "starting E step on doc {0}".format(d)
    graphlib.initialize_random(phi)

    ensure(phi.shape[1] == beta.shape[0] == len(gamma) == len(alpha))
    ensure(phi.shape[0] == len(document))
    ensure(len(eta) < phi.shape[1]) # is partial

    i = 0
    min_iter = 20 - global_iteration
    max_iter = last_local_iterations if last_local_iterations > 0 else 20
    last_local_elbo, local_elbo = graphlib.INITIAL_ELBO - 100, graphlib.INITIAL_ELBO
    while graphlib.elbo_did_not_converge(local_elbo, last_local_elbo, i, 
                                        criterion=0.01,
                                        min_iter=min_iter, max_iter=max_iter):
        #print 'will update gamma...'
        # update gammas
        lda_update_gamma(alpha, phi, gamma)

        #print 'will update phis...'
        ensure(len(eta) < phi.shape[1]) #otherwise it's a full update
        partial_slda_update_phi(document, phi, gamma, beta, y, eta, sigma_squared)

        # speed things up by maxing out in first five E runs
        # also use same as last local iterations
        if last_local_iterations == 0:
            #print 'will calculate elbo...'
            # calculate new ELBO
            last_local_elbo = local_elbo
            local_elbo = partial_slda_local_elbo(document, y, 
                                                    alpha, beta, gamma, phi, 
                                                    eta, sigma_squared)
        i += 1

        #print {'beta': beta, 'gamma': gamma, 'phi': phi, 'y': y, 'eta': eta}
        #print "{2}: e-step iteration {0} ELBO: {1}".format(i, local_elbo, global_iteration)
    if d % 100 == 0:
        print "{2}: done pSLDA e-step on doc {3}: {0} iterations ELBO: {1}".format(i, local_elbo, global_iteration, d)
    return i

def slda_E_step_for_doc(global_iteration, 
                        last_local_iterations,
                        d, document, y,
                        alpha, beta, gamma, phi,
                        eta, sigma_squared):
    """Given phi and gamma matrices and document of the document.
        Recalculate phi and gamma repeatedly iteratively.
        Also recalculate y.
        Uses local elbo calculation to check for convergence.
    """
    print "starting E step on doc {0}".format(d)
    graphlib.initialize_random(phi)

    i = 0
    max_iter = last_local_iterations if last_local_iterations > 0 else 20
    last_local_elbo, local_elbo = graphlib.INITIAL_ELBO - 100, graphlib.INITIAL_ELBO
    while graphlib.elbo_did_not_converge(local_elbo, last_local_elbo, i, 
                                            criterion=0.01, max_iter=max_iter):
        #print 'will update gamma...'
        # update gammas
        lda_update_gamma(alpha, phi, gamma)

        #print 'will update phis...'
        slda_update_phi(document, phi, gamma, beta, y, eta, sigma_squared)

        # speed things up by maxing out in first five E runs
        # also use same as last local iterations
        if last_local_iterations == 0:
            #print 'will calculate elbo...'
            # calculate new ELBO
            last_local_elbo = local_elbo
            local_elbo = slda_local_elbo(document, y, 
                                         alpha, beta, gamma, phi, 
                                         eta, sigma_squared)
        i += 1

        #print {'beta': beta, 'gamma': gamma, 'phi': phi, 'y': y, 'eta': eta}
        #print "{2}: e-step iteration {0} ELBO: {1}".format(i, local_elbo, global_iteration)
    print "{2}: done e-step on doc {3}: {0} iterations ELBO: {1}".format(i, local_elbo, global_iteration, d)
    return i


def lm_E_step_for_doc(global_iteration,
                        d, document, comment, 
                        alphaD, alphaC, 
                        betaD, betaC, 
                        gammaD, gammaC, 
                        phiD, phiC, 
                        y, eta, sigma_squared):
    """Given phi and gamma matrices and document of the document.
        Recalculate phi and gamma repeatedly iteratively.
        Uses local elbo calculation to check for convergence.
    """
    print "starting E step on doc {0}".format(d)
    graphlib.initialize_random(phiD)
    graphlib.initialize_random(phiC)

    i = 0
    last_local_elbo, local_elbo = graphlib.INITIAL_ELBO - 100, graphlib.INITIAL_ELBO
    while graphlib.elbo_did_not_converge(local_elbo, last_local_elbo, i, 
                                            criterion=0.1, max_iter=20):
        print 'will update gamma...'
        # update gammas
        lda_update_gamma(alphaD, phiD, gammaD)
        lda_update_gamma(alphaC, phiC, gammaC)

        Nd,Kd = phiD.shape

        print 'will update phis...'
        # update phis (note we have to pass the right part of eta!)
        slda_update_phi(document, phiD, gammaD, betaD, y[d], eta[:Kd], sigma_squared)
        slda_update_phi(comment, phiC, gammaC, betaC, y[d], eta[Kd:], sigma_squared)

        print 'will calculate y...'
        # update the response variable
        # y = ηTE[Z] = ηTφ      [  where φ = 1/N * Σnφn   ]
        y[d] = np.dot(eta, calculate_EZ_from_small_phis(phiD, phiC))

        if i % 2 == 0:
            print 'will calculate elbo...'
            # calculate new ELBO
            last_local_elbo = local_elbo
            local_elbo = lm_local_elbo(document, comment, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y[d], eta, sigma_squared)
        i += 1

        #print {'beta': (betaD, betaC), 'gamma': (gammaD, gammaC), 'phi': (phiD, phiC), 'y': y, 'eta': eta}
        print "{2}: e-step iteration {0} ELBO: {1}".format(i, local_elbo, global_iteration)
    print "{2}: done e-step on doc {3}: {0} iterations ELBO: {1}".format(i, local_elbo, global_iteration, d)
    return i

def iterwords(text):
    """Accepts an list of 2-tuples.
        Yields a generator of n,word,count triplets to keep track
        of which word corresponds to the "nth" word in each document.
    """
    n = 0
    for word,count in text:
        for i in xrange(count):
            yield n,word,count
            n += 1

def lm_local_elbo(document, comment,
                         alphaD, alphaC, 
                         betaD, betaC, 
                         gammaD, gammaC, 
                         phiD, phiC, 
                         y, eta, sigma_squared):
    """Given all of the parametes for one document.
        Calculate the evidence lower bound.
        Helps you know when convergence happens in E step.

   ELBO = ℒ = 
    E[log p(θD|αD)] + ΣNE[log p(ZnD|θD)] + ΣNE[log p(wnD|ZnD,β1:KD)]
    E[log p(θC|αC)] + ΣNE[log p(ZnC|θC)] + ΣNE[log p(wnD|ZnD,β1:KD)] + 
    E[log p(y|Z1:N,η,σ2)] + H(q)

    The first 3 and second 3 are LDA terms 
        for document and comment respectively.:

    The last terms are for the y, similar to as in sLDA, 
        and for the entropy of q.
    """
    elbo = 0.0
    #print 'elbo lda terms...'
    elbo += lda_elbo_terms(document, alphaD, betaD, gammaD, phiD)
    elbo += lda_elbo_terms(comment, alphaC, betaC, gammaC, phiC)

    #print 'elbo slda y...'
    elbo += lm_elbo_y_from_small_phis(y, eta, phiD, phiC, sigma_squared)

    #print 'elbo entropy...'
    elbo += lm_elbo_entropy(gammaD, gammaC, phiD, phiC)
    return elbo

def lm_elbo_entropy(gammaD, gammaC, phiD, phiC):
    """Calculates entropy of the variational distribution q.

    H(q) = 
    – ΣNΣK φDn,klog φDn,k – log Γ(ΣKγkD) + ΣKlog Γ(γkD)  – ΣK(γkD – 1)E[log θkD]
    – ΣNΣK φCn,klog φCn,k – log Γ(ΣKγkC) + ΣKlog Γ(γkC)  – ΣK(γkC – 1)E[log θkC]
    """
    elbo = 0.0
    elbo += lda_elbo_entropy(gammaD, phiD)
    elbo += lda_elbo_entropy(gammaC, phiC)
    return elbo


def lda_elbo_entropy(gamma, phi):
    """Entropy of variational distribution q in LDA.

    Accepts phi (N x K) matrix.
            gamma (a K-size vector) for document


    Returns double representing the entropy in the elbo of LDA..

    H(q) = 
    – ΣNΣK φDn,klog φDn,k – log Γ(ΣKγkD) + ΣKlog Γ(γkD)  – ΣK(γkD – 1)E[log θkD]
    """
    elbo = 0.0
    (N,K) = phi.shape
    ensure(len(gamma) == K)
    elbo += -1 * np.sum(phi * np.log(phi))

    elbo += -1 * gammaln(np.sum(gamma))
    elbo += np.sum(gammaln(gamma))

    ElogTheta = graphlib.dirichlet_expectation(gamma)
    ensure(ElogTheta.shape == gamma.shape)
    elbo += -1 * sum((gamma - 1) * ElogTheta)

    return elbo


def lm_elbo_y_from_small_phis(y, eta, phiD, phiC, sigma_squared):
    """
    Calculates some terms in the elbo for a document.
    Same as in sLDA.

    E[log p(y|Z1:N,η,σ2)] = (–1/2)log 2πσ2 – (1/2σ2)[y2– 2yηTE[Z] + ηTE[ZZT]η]

    Test:
    Should be the same as slda_elbo_y when phiD and phiC are catercorner concatenated.
    """
    elbo = 0.0
    ss = sigma_squared
    elbo += (-0.5) * np.log(2 * np.pi * ss)
    
    ez = calculate_EZ_from_small_phis(phiD, phiC)
    ezzt = calculate_EZZT_from_small_phis(phiD, phiC)
    nEZZTn = np.dot(np.dot(eta, ezzt), eta)
    elbo += (-0.5 / ss) * (y*y - (2 * y * np.dot(eta, ez)) + nEZZTn)
    return elbo

def slda_elbo_y(y, eta, phi, sigma_squared):
    """
    Calculates some terms in the elbo for a document.
    Same as in sLDA.

    E[log p(y|Z1:N,η,σ2)] = (–1/2)log 2πσ2 – (1/2σ2)[y2– 2yηTE[Z] + ηTE[ZZT]η]
    """
    elbo = 0.0
    ss = sigma_squared
    elbo += (-0.5) * np.log(2 * np.pi * ss)
    
    #print 'will calculate ez...'
    ez = calculate_EZ(phi)
    #print 'will calculate ezzt...'
    ezzt = calculate_EZZT(phi)
    #print 'will calculate nEZZTn...'
    nEZZTn = np.dot(np.dot(eta, ezzt), eta)
    #print 'will sum up elbo...'
    elbo += (-0.5 / ss) * (y*y - (2 * y * np.dot(eta, ez)) + nEZZTn)
    return elbo
    

def lda_elbo_terms(document, alpha, beta, gamma, phi):
    """
    Calculates some terms in the elbo for a document.
    Same as in LDA.

    E[log p(θD|αD)] + ΣNE[log p(ZnD|θD)] + ΣNE[log p(wnD|ZnD,β1:KD)]

    E[log p(θ|a)] = log Γ(Σkai) – Σklog Γ(ai) + ΣK(ak-1)E[log θk] 
    E[log p(Zn|θ)] = ΣKφn,kE[log θk]
    E[log p(wn|Zn,β1:K)]  = ΣKφn,klog βk,Wn

    (Note that E[log θk] = Ψ(γk) – Ψ(Σj=1..Kγj) ).
    """
    N,K = phi.shape
    elbo = 0.0

    # E[log p(θ|a)] = log Γ(Σkai) – Σklog Γ(ai) + ΣK(ak-1)E[log θk] 
    elbo += gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

    ElogTheta = graphlib.dirichlet_expectation(gamma)
    #assert len(ElogTheta) == len(alpha)
    #assert ElogTheta.shape == alpha.shape
    elbo += np.sum((alpha - 1) * ElogTheta)

    if isinstance(document, np.ndarray):
        # even faster optimization
        elbo += np.sum(phi * (ElogTheta + (np.log(beta[:,document]).T)))
    else:
        for n,word,count in iterwords(document):
            # E[log p(Zn|θ)] = ΣKφn,kE[log θk]
            # E[log p(wn|Zn,β1:K)]  = ΣKφn,klog βk,Wn

            # optimization:
            # E[log p(Zn|θ)] + E[log p(wn|Zn,β1:K)] = ΣKφn,k(E[log θk] + log βk,Wn)
            elbo += np.sum(phi[n] * (ElogTheta + np.log(beta[:,word])))

    return elbo


def lm_global_elbo(documents, comments, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y, eta, sigma_squared):
    """Given all of the parametes.
        Calculate the evidence lower bound.
        Helps you know when convergence happens.
    """
    return np.sum(lm_local_elbo(documents[d], comments[d], alphaD, alphaC, betaD, betaC, gammaD[d], gammaC[d], phiD[d], phiC[d], y[d], eta, sigma_squared) for d in xrange(len(documents)))

def read_sparse(filename):
    """Accepts filename.
        Reads in sparse data in wordid:count form, 
        one line is one doc.
    """
    docs = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            line = l.strip('\r\n ')
            if line != '':
                doc = [(int(w.split(':')[0]),int(w.split(':')[1])) 
                            for w in line.split(' ')]
                docs.append(doc)
    return docs
