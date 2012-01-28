#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
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


########### PYPY-only functions! ###########
def ispypy():
    """Returns a boolean True if pypy is running the program.
        Does this by checking the matrix module, which is not currently implemented.
    """
    return True
    return 'matrix' in dir(np)

def axis_sum(matrix, axis):
    """Accepts a 2-d array and axis integer (0 or 1).
    """
    assert axis in [0, 1]
    if ispypy():
        nrows,ncols = matrix.shape
        if axis == 1:
            return np.array([np.sum(matrix[i]) for i in xrange(nrows)])
        else:
            return np.array([np.sum(matrix[:,i]) for i in xrange(ncols)])
    else:
        rowsums = np.sum(matrix, axis=1)
        return rowsums

def np_concatenate((a, b), axis=1):
    """Accepts a 1-d array and axis (only 1) and concatenates them.
    """
    assert axis in [1]
    if ispypy():
        n = np.zeros((len(a) + len(b),))
        n[:len(a)] = a
        n[len(a):] = b
        return n
    else:
        return np.concatenate(a, b, axis=1)


def matrix_multiply(a, b):
    """Takes two matrices and does a complicated matrix multiply.  Yes that one.
    """
    if len(a.shape) == 1:
        nrows, = a.shape
        a = np.zeros((nrows, 1))

    if len(b.shape) == 1:
        bc, = b.shape
        if bc == a.shape[1]:
            b = np.zeros((bc, 1))
        else:
            b = np.zeros((1, bc))
       
    nrows,ac = a.shape
    bc,ncols = b.shape

    assert ac == bc
    if ispypy():
        n = np.zeros((nrows, ncols))
        for i in xrange(nrows):
            for j in xrange(ncols):
                n[i,j] = np.sum(a[i] * b[:,j])
        return n
    else:
        np.dot(a, b)


def np_diag(a):
    """Takes a 1-d vector and makes it the diagonal of a 2-d vector
    """
    if ispypy():
        nrows, = a.shape
        n = np.zeros((nrows, nrows))
        for i in xrange(nrows):
            n[i,i] = a[i]
        return n
    else:
        return np.diag(a)

def np_second_arg_array_index(matrix, array):
    """look at second part."""
    if ispypy():
        nrows,ncols = matrix.shape
        if len(array.shape) == 1:
            n = np.zeros((1, array.shape[0]))
            for i in xrange(array.shape[0]):
                n[0,i] = np.sum(matrix[:,int(array[i])])
            return n
        else:
            assert len(array.shape) == 2
            n = np.zeros(array.shape)
            for i in xrange(array.shape[0]):
                n[i] = np_second_arg_array_index(matrix, array[i])
            return n

    else:
        return matrix[:,array]

def np_log(a):
    """Takes a nd array or int, returns log
    """
    if ispypy():
        if isinstance(a, np.ndarray):
            n = np.zeros(a.shape)
            if len(a.shape) == 1:
                for i in xrange(len(a)):
                    n[i] = math.log(a[i])
            else:
                assert len(a.shape) == 2
                for i in xrange(a.shape[0]):
                    for j in xrange(a.shape[1]):
                        n[i,j] = math.log(a[i,j])
            return n
        else:
            return math.log(a)
    else:
        return np.log(a)
########### END END PYPY-only functions! ###########



INITIAL_ELBO = float('-inf')

final_output = {}


def logsumexp(a, axis=0):
    """Same as sum, but in log space. Compare to logaddexp."""
    return np.logaddexp.reduce(a, axis)

def log_row_normalize(m):
    """Does row-normalize in log space.
        All values in m are log probabilities.
    """
    assert len(m.shape) == 2
    lognorm = logsumexp(m, axis=1)
    lognorm.shape = (lognorm.shape[0], 1)

    m -= lognorm
    return m

def dirichlet_expectation(alpha):
    """
    From Matt Hoffman:
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    #assert len(alpha.shape) == 1 # jperla: not sure what else it does
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    else:
        return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def row_normalize(matrix):
    """Accepts 2-D matrix.
        Modifies matrix in place.   
        Returns matrix with rows normalized.
    """
    nrows, ncols = matrix.shape
    rowsums = axis_sum(matrix, axis=1)

    if ispypy():
        for i in xrange(len(rowsums)):
            matrix[i] /= rowsums[i]
    else:
        np.divide(matrix, np.matrix(rowsums).T, matrix)
    return matrix

def initialize_log_beta(num_topics, num_words):
    """Initializes beta randomly using a random dirichlet.
        Accepts integers number of topics, and number of words in vocab.
        Returns a TxW matrix which have the probabilities of word
            distributions.  Exp of each row sums to 1.
    """
    l = 1*np.random.gamma(100., 1./100., (num_topics, num_words))
    Elogbeta = dirichlet_expectation(l)
    beta = log_row_normalize(Elogbeta)
    return beta

def initialize_beta(num_topics, num_words):
    """Initializes beta randomly using a random dirichlet.
        Accepts integers number of topics, and number of words in vocab.
        Returns a TxW matrix which have the probabilities of word
            distributions.  Each row sums to 1.
    """
    log_beta = initialize_log_beta(num_topics, num_words)
    return np.exp(log_beta)

def initialize_uniform(matrix):
    """Accepts a matrix with a defined shape.
        Initializes it to to be uniform probability on row.
        Each row on last dimension should sum to 1.
        Returns the original matrix, modified.
    """
    nrows,ncols = matrix.shape
    matrix = np.ones(matrix.shape)*(1.0/ncols)
    return matrix

def initialize_log_uniform(matrix):
    """Returns log of initialize_uniform."""
    if len(matrix.shape) == 1:
        matrix[:] = np.log(initialize_uniform(matrix))
    else:
        matrix[:,:] = np.log(initialize_uniform(matrix))
    return matrix

def random_sample(shape):
    a = np.zeros(shape)
    for i in xrange(len(a)):
        a[i] = random.random()
    return row_normalize(a)

def initialize_random(matrix):
    """Accepts a matrix with a defined shape.
        Initializes it to to random probabilities on row.
        Each row on last dimension should sum to 1.
        Returns the original matrix, modified.
    """
    if matrix.ndim == 2:
        matrix[:,:] = random_sample(matrix.shape)
        row_normalize(matrix)
    else:
        # one-dimensional array
        matrix[:] = random_sample(matrix.shape)
        matrix[:] = matrix / sum(matrix)
    return matrix

def initialize_log_random(matrix):
    """Returns log of initialize_random."""
    if len(matrix.shape) == 1:
        matrix[:] = np.log(initialize_random(matrix))
    else:
        matrix[:,:] = np.log(initialize_random(matrix))
    return matrix

def elbo_did_not_converge(elbo, last_elbo, num_iter=0, 
                          criterion=0.001, max_iter=20):
    """Accepts two elbo doubles.  
        Also accepts the number of iterations already performed in this loop.
        Also accepts convergence criterion: 
            (elbo - last_elbo) < criterion # True to stop
        Finally, accepts 
        Returns boolean.
        Figures out whether the elbo is sufficiently smaller than
            last_elbo.
    """
    if num_iter >= max_iter:
        return False

    if elbo == INITIAL_ELBO or last_elbo == INITIAL_ELBO:
        return True
    else:
        # todo: do a criterion convergence test
        if np.abs(elbo - last_elbo) < criterion:
            return False
        else:
            return True

def recalculate_beta(text, beta, phi):
    """
    update topics: βk,wnew ∝ ΣdΣn 1(wd,n = w) φkd,n

    Accepts beta matrix (KxW) and 
        phi, a D-length list of (N x K) matrices.
    """
     # todo: logarithms?
    (K,W) = beta.shape
    D = len(phi)
    beta[:,:] = np.zeros(beta.shape)    

    if isinstance(text[0], np.ndarray):
        for d in xrange(D):
            for n,word in enumerate(text[d]):
                for k in xrange(K):
                    beta[k,word] += phi[d][n][k]
    else:
        for d in xrange(D):
            for n,word,count in iterwords(text[d]):
                for k in xrange(K):
                    beta[k,word] += phi[d][n][k]
    row_normalize(beta)
    return beta

def recalculate_log_beta(text, log_beta, log_phi):
    """
    update topics: βk,wnew ∝ ΣdΣn 1(wd,n = w) φkd,n

    Accepts log beta matrix (KxW) and 
        log phi, a D-length list of (N x K) matrices.
    """
     # todo: logarithms?
    (K,W) = log_beta.shape
    D = len(log_phi)

    # todo: jperla: should use -inf or a different really small number?!
    log_beta[:,:] = np.ones(log_beta.shape) * float('-300')
    
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
    log_row_normalize(log_beta)
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


def recalculate_eta_sigma(eta, y, phi1, phi2):
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
    assert len(phi1) == len(phi2)
    D = len(phi1)

    Nd,K = phi1[0].shape
    Nc,J = phi2[0].shape
    Ndc, KJ = (Nd+Nc,K+J)

    #print 'e_a...'
    E_A = np.zeros((D, KJ))
    for d in xrange(D):
        E_A[d,:] = calculate_EZ_from_small_phis(phi1[d], phi2[d])
  
    #print 'inverse...'
    E_ATA_inverse = calculate_E_ATA_inverse(phi1, phi2)

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
    Ndc,KJ = big_phi.shape
    return np.sum(big_phi, axis=0) / Ndc

    
def calculate_EZ_from_big_log_phi(big_log_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc,KJ = big_log_phi.shape
    return logsumexp(big_log_phi, axis=0) - np.log(Ndc)

def logdotexp(a, b):
    """Accepts two matrices a, b.
        The matrices represent matrices with log probabilities.
        Returns np.log(np.dot(np.exp(a), np.exp(b))).

        Log of dot product of real probabilities.
        Does all computations in log space.

        Cmompare to np.dot().
    """
    (a1,a2) = a.shape
    (b1,b2) = b.shape
    assert a2 == b1
    output = np.zeros((a1, b2))
    for i in xrange(a1):
        for j in xrange(b2):
            output[i,j] = logsumexp([(x + y) for x,y in izip(a[i,:], b[:,j])])
    return output

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
            inner_sum[i,j] = np.sum(m) - np.sum(np.diag(m))

    for i in xrange(J):
        for j in xrange(J):
            m = np.dot(np.matrix(p2[:,i]), np.matrix(p2[:,j]).T)
            inner_sum[K+i,K+j] = np.sum(m) - np.sum(np.diag(m))

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
    assert big_phi_sum.shape == (KJ,)
    inner_sum += np.diag(big_phi_sum)

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
            m += np.diag(np.ones(Nd) * -1000)
            inner_sum[i,j] = logsumexp(m.flatten())

    for i in xrange(J):
        for j in xrange(J):
            m = logdotexp(np.matrix(p2[:,i]), np.matrix(p2[:,j]).T)
            m += np.diag(np.ones(Nc) * -1000)
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
    assert big_phi_sum.shape == (KJ,)
    for i in xrange(KJ):
        inner_sum[i,i] = logsumexp([inner_sum[i,i], big_phi_sum[i]])

    inner_sum -= np.log(Ndc * Ndc)
    return inner_sum

def calculate_EZZT(big_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[ZdZdT].
        Returns the final matrix ((K+J) x (K+J)).

        (Also, E[ZdZdT] = (1/N2)(ΣNΣm!=nφd,nφd,mT  +  ΣNdiag{φd,n})
    """
    (Ndc, KJ) = big_phi.shape
    inner_sum = np.zeros((KJ, KJ))
    for n in xrange(Ndc):
        for m in xrange(Ndc):
            if n != m:
                #new_matrix = matrix_multiply(big_phi[n].T, big_phi[m])
                new_matrix = np.dot(np.matrix(big_phi[n]).T, np.matrix(big_phi[m]))
                assert new_matrix.shape == inner_sum.shape
                inner_sum += new_matrix
    inner_sum += np.diag(np.sum(big_phi, axis=0))

    inner_sum /= (Ndc * Ndc)
    return inner_sum

def calculate_E_ATA_inverse(phi1, phi2):
    """Accepts number of documents, 
        and a big_phi matrix of size (Nd + Nc, K + J).
        Returns a new matrix which is inverse of E([ATA]) of size (K+J,K+J).

        (Note that A is the D X (K + J) matrix whose rows are the vectors ZdT for document and comment concatenated.)
        (Also note that the dth row of E[A] is φd, and E[ATA] = Σd E[ZdZdT] .)
    """
    D = len(phi1)
    Nd,K = phi1[0].shape
    Nc,J = phi2[0].shape
    (Ndc, KJ) = (Nd+Nc, K+J)
    E_ATA = sum(calculate_EZZT_from_small_phis(phi1[d], phi2[d]) for d in xrange(D))
    assert E_ATA.shape == (KJ, KJ)

    if ispypy():
        # todo: this is massively broken!!!
        return np_diag(np.ones((KJ,)))
    else:
        return np.linalg.inv(E_ATA)

def update_gamma_lda_E_step(alpha, phi, gamma):
    """
     Accepts:
        gamma and alpha are K-size vectors.
        Phi is an NxK vector.
     Returns gamma.

     update gamma: γnew ← α + Σnφn
    """
    assert phi.shape[1] == len(gamma)
    gamma[:] = alpha + axis_sum(phi, axis=0)
    return gamma


def update_gamma_lda_E_step_from_log(log_alpha, log_phi, log_gamma):
    """
     Same as update_gamma_lda_E_step, 
        but in log probability space.
    """
    assert log_phi.shape[1] == len(log_gamma)
    log_gamma[:] = logsumexp([log_alpha, logsumexp(log_phi, axis=0)], axis=0)
    return log_gamma


def _unoptimized_update_phi_lda_E_step(text, phi, gamma, beta, y_d, eta, sigma_squared):
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

    phi_sum = axis_sum(phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = dirichlet_expectation(gamma)
    assert len(ElogTheta) == K

    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = (eta * eta)
    front = (-1.0 / (2 * N * Ns))

    for n,word,count in iterwords(text):
        phi_sum -= phi[n]
        assert len(phi_sum) == K

        pB = np_log(beta[:,word])
        pD = (front * (((2 * np.dot(eta, phi_sum) * eta) + eta_dot_eta))
                            )
        assert len(pB) == K
        assert len(pC) == K
        assert len(pD) == K

        # must exponentiate and sum immediately!
        phi[n,:] = np.exp(ElogTheta + pB + pC + pD)
        phi[n,:] /= np.sum(phi[n,:])

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

def update_phi_lda_E_step(text, phi, gamma, beta, y_d, eta, sigma_squared):
    """
        Update phi in LDA. 
        phi is N x K matrix.
        gamma is a K-size vector

     update phid:
     φd,n ∝ exp{ E[log θ|γ] + 
                 E[log p(wn|β1:K)] + 
                 (y / Nσ2) η  — 
                 [2(ηTφd,-n)η + (η∘η)] / (2N2σ2) }
            exp{ A + B + C - D}
     
     Note that E[log p(wn|β1:K)] = log βTwn
    """
    (N, K) = phi.shape

    phi_sum = np.sum(phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = dirichlet_expectation(gamma)

    front = (-1.0 / (2 * N * Ns))
    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = front * (eta * eta)
    const = ElogTheta + pC + eta_dot_eta

    right_eta_times_const = (front * 2 * eta)

    if isinstance(text, np.ndarray):
        # if text is in array form, do an approximate fast matrix update
        phi_minus_n = -(phi - phi_sum)
        #phi[:,:] = np_log(np_second_arg_array_index(beta,text).T)
        phi[:,:] = np.log(beta[:,text].T)
        #phi[:,:] += matrix_multiply(axis_sum(eta * phi_minus_n, axis=1).T, right_eta_times_const)
        phi[:,:] += np.dot(np.matrix(np.dot(phi_minus_n, eta)).T, np.matrix(right_eta_times_const))
        phi[:,:] += const
        phi[:,:] = np.exp(phi[:,:])
        row_normalize(phi)
    else:
        # otherwise, iterate through each word
        for n,word,count in iterwords(text):
            phi_sum -= phi[n]

            pB = np_log(beta[:,word])
            pD = (np.dot(eta, phi_sum) * right_eta_times_const) 

            # must exponentiate and normalize immediately!
            phi[n,:] = np.exp(pB + pD + const)
            phi[n,:] /= np.sum(phi[n,:])

            # add this back into the sum
            # unlike in LDA, this cannot be computed in parallel
            phi_sum += phi[n]
    return phi

def update_log_phi_lda_E_step(text, log_phi, log_gamma, log_beta, y_d, eta, sigma_squared):
    """
        Same as update_phi_lda_E_step but in log probability space.
    """
    (N, K) = log_phi.shape

    log_phi_sum = logsum(log_phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = dirichlet_expectation(np.exp(log_gamma))

    front = (-1.0 / (2 * N * Ns))
    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = front * (eta * eta)
    const = ElogTheta + pC + eta_dot_eta

    right_eta_times_const = (front * 2 * eta)

    if isinstance(text, np.ndarray):
        # if text is in array form, do an approximate fast matrix update
        log_phi_minus_n = -1 + (logsumexp([log_phi, (-1 + log_phi_sum)]))
        phi[:,:] = log_beta[:,text].T

        phi[:,:] = logsumexp(phi[:,:], logdotexp(np.matrix(logdotexp(log_phi_minus_n, np.log(eta))).T, np.matrix(np.log(right_eta_times_const))))
        phi[:,:] = logsumexp(phi[:,:], np.log(const))

        log_row_normalize(phi)
    else:
        # otherwise, iterate through each word
        for n,word,count in iterwords(text):
            phi_sum -= phi[n]

            pB = np_log(beta[:,word])
            pD = (np.dot(eta, phi_sum) * right_eta_times_const) 

            # must exponentiate and normalize immediately!
            phi[n,:] = np.exp(pB + pD + const)
            phi[n,:] /= np.sum(phi[n,:])

            # add this back into the sum
            # unlike in LDA, this cannot be computed in parallel
            phi_sum += phi[n]
    return phi

def do_E_step(global_iteration,
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
    (Nd,Kd) = phiD.shape
    (Nc,Kc) = phiC.shape

    initialize_random(phiD)
    initialize_random(phiC)

    local_elbo, local_last_elbo = 0, 0
    print "starting E step on doc {0}".format(d)
    i = 0

    local_elbo = INITIAL_ELBO
    last_local_elbo = INITIAL_ELBO - 100
    while elbo_did_not_converge(local_elbo, last_local_elbo, i, criterion=0.1, max_iter=20):
        print 'will update gamma...'
        # update gammas
        update_gamma_lda_E_step(alphaD, phiD, gammaD)
        update_gamma_lda_E_step(alphaC, phiC, gammaC)

        print 'will update phis...'
        # update phis (note we have to pass the right part of eta!)
        update_phi_lda_E_step(document, phiD, gammaD, betaD, y[d], eta[:Kd], sigma_squared)
        update_phi_lda_E_step(comment, phiC, gammaC, betaC, y[d], eta[Kd:], sigma_squared)

        print 'will calculate y...'
        # update the response variable
        # y = ηTE[Z] = ηTφ      [  where φ = 1/N * Σnφn   ]
        y[d] = np.dot(eta, calculate_EZ_from_small_phis(phiD, phiC))

        if i % 2 == 0:
            print 'will calculate elbo...'
            # calculate new ELBO
            last_local_elbo = local_elbo
            local_elbo = calculate_local_elbo(document, comment, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y[d], eta, sigma_squared)
        '''
        '''
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

def calculate_local_elbo(document, comment,
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
    elbo += elbo_lda_terms(alphaD, gammaD, phiD, betaD, document)
    elbo += elbo_lda_terms(alphaC, gammaC, phiC, betaC, comment)

    #print 'elbo slda y...'
    elbo += elbo_slda_y(y, eta, phiD, phiC, sigma_squared)

    #print 'elbo entropy...'
    elbo += elbo_entropy(gammaD, gammaC, phiD, phiC)
    return elbo

def elbo_entropy(gammaD, gammaC, phiD, phiC):
    """Calculates entropy of the variational distribution q.

    H(q) = 
    – ΣNΣK φDn,klog φDn,k – log Γ(ΣKγkD) + ΣKlog Γ(γkD)  – ΣK(γkD – 1)E[log θkD]
    – ΣNΣK φCn,klog φCn,k – log Γ(ΣKγkC) + ΣKlog Γ(γkC)  – ΣK(γkC – 1)E[log θkC]
    """
    elbo = 0.0
    elbo += elbo_entropy_lda(phiD, gammaD)
    elbo += elbo_entropy_lda(phiC, gammaC)
    return elbo

def elbo_entropy_lda(phi, gamma):
    """Entropy of variational distribution q in LDA.

    Accepts phi (N x K) matrix.
            gamma (a K-size vector) for document


    Returns double representing the entropy in the elbo of LDA..

    H(q) = 
    – ΣNΣK φDn,klog φDn,k – log Γ(ΣKγkD) + ΣKlog Γ(γkD)  – ΣK(γkD – 1)E[log θkD]
    """
    elbo = 0.0
    (N,K) = phi.shape
    assert len(gamma) == K
    elbo += -1 * np.sum(phi * np_log(phi))

    elbo += -1 * gammaln(np.sum(gamma))
    elbo += np.sum(gammaln(gamma))

    ElogTheta = dirichlet_expectation(gamma)
    assert ElogTheta.shape == gamma.shape
    elbo += -1 * sum((gamma - 1) * ElogTheta)

    return elbo


def elbo_slda_y(y, eta, phiD, phiC, sigma_squared):
    """
    Calculates some terms in the elbo for a document.
    Same as in sLDA.

    E[log p(y|Z1:N,η,σ2)] = (–1/2)log 2πσ2 – (1/2σ2)[y2– 2yηTE[Z] + ηTE[ZZT]η]
    """
    elbo = 0.0
    ss = sigma_squared
    elbo += (-0.5) * np_log(2 * np.pi * ss)
    
    ez = calculate_EZ_from_small_phis(phiD, phiC)
    ezzt = calculate_EZZT_from_small_phis(phiD, phiC)
    nEZZTn = np.dot(np.dot(eta, ezzt), eta)
    elbo += (-0.5 / ss) * (y*y - (2 * y * np.dot(eta, ez)) + nEZZTn)
    return elbo
    

def elbo_lda_terms(alpha, gamma, phi, beta, document):
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

    ElogTheta = dirichlet_expectation(gamma)
    #assert len(ElogTheta) == len(alpha)
    #assert ElogTheta.shape == alpha.shape
    elbo += np.sum((alpha - 1) * ElogTheta)

    if isinstance(document, np.ndarray):
        # even faster optimization
        elbo += np.sum(phi * (ElogTheta + (np_log(np_second_arg_array_index(beta,document).T))))
    else:
        for n,word,count in iterwords(document):
            # E[log p(Zn|θ)] = ΣKφn,kE[log θk]
            # E[log p(wn|Zn,β1:K)]  = ΣKφn,klog βk,Wn

            # optimization:
            # E[log p(Zn|θ)] + E[log p(wn|Zn,β1:K)] = ΣKφn,k(E[log θk] + log βk,Wn)
            elbo += np.sum(phi[n] * (ElogTheta + np_log(beta[:,word])))

    return elbo


def calculate_global_elbo(documents, comments, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y, eta, sigma_squared):
    """Given all of the parametes.
        Calculate the evidence lower bound.
        Helps you know when convergence happens.
    """
    return np.sum(calculate_local_elbo(documents[d], comments[d], alphaD, alphaC, betaD, betaC, gammaD[d], gammaC[d], phiD[d], phiC[d], y[d], eta, sigma_squared) for d in xrange(len(documents)))

            

