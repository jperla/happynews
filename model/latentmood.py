#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, time, string
from itertools import izip,product,repeat,chain

import numpy as ny
from scipy.special import psi
from scipy.special import gamma as GAMMA


INITIAL_ELBO = -1e10

def dirichlet_expectation(alpha):
    """
    From Matt Hoffman:
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(ny.sum(alpha)))
    else:
        return(psi(alpha) - psi(ny.sum(alpha, 1))[:, ny.newaxis])

def row_normalize(matrix):
    """Accepts 2-D matrix.
        Modifies matrix in place.   
        Returns matrix with rows normalized.
    """
    nrows, ncols = matrix.shape
    rowsums = ny.sum(matrix, axis=1)
    ny.divide(matrix, rowsums.reshape(nrows, 1), matrix)
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
    nrows,ncols = matrix.shape
    matrix = ny.ones(matrix.shape)*(1.0/ncols)
    return matrix

def initialize_random(matrix):
    """Accepts a matrix with a defined shape.
        Initializes it to to random probabilities on row.
        Each row on last dimension should sum to 1.
        Returns the original matrix, modified.
    """
    if matrix.ndim == 2:
        matrix[:,:] = ny.random.sample(matrix.shape)
        row_normalize(matrix)
    else:
        matrix[:] = ny.random.sample(matrix.shape)
        matrix[:] = matrix / sum(matrix)
    return matrix

def elbo_did_not_converge(elbo, last_elbo):
    """Takes two elbo doubles.  
        Returns boolean.
        Figures out whether the elbo is sufficiently smaller than
            last_elbo.
    """
    if elbo == INITIAL_ELBO or last_elbo == INITIAL_ELBO:
        return True
    else:
        # todo: do a criterion convergence test
        if ny.abs(elbo - last_elbo) < 0.00001:
            return False
        else:
            return True

def recalculate_beta(text, beta, phi):
    """
    update topics: βk,wnew ∝ ΣdΣn 1(wd,n = w) φkd,n

    Accepts beta matrix (KxW) and 
        phi, a dictionary of (N x K) matrices.
    """
     # todo: logarithms?
    (K,W) = beta.shape
    D = len(phi)
    for k,w in product(xrange(K), xrange(W)):
        beta[k,w] = 0
        for d in xrange(D):
            for n,word,count in iterwords(text[d]):
                if word == w:
                    beta[k,w] += phi[d][n][k]
    row_normalize(beta)
    return beta

def run_em(data):
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

    vocab = set(chain(*[d.keys() for d in documents]))
    vocab.update(chain(*[c.keys() for c in comments]))
    W = len(vocab)
    D = len(documents)
    K = 2
    J = 3

    assert D > 0

    # calculate number of words per document/comment
    document_Nds = [sum(d.values()) for d in documents]
    comment_Nds = [sum(c.values()) for c in comments]

    # "it suffices to fix alpha to uniform 1/K"
    alphaD = ny.ones((K,)) * (1.0 / K)
    alphaC = ny.ones((J,)) * (1.0 / J)

    # Initialize the variational distribution q(beta|lambda)
    betaD = initialize_beta(K, W)
    betaC = initialize_beta(J, W)

    phiD = dict((d,ny.ones((document_Nds[d], K))*(1.0/K)) for d in xrange(D))
    gammaD = ny.ones((D, K)) * (1.0 / K)
    initialize_random(gammaD)

    phiC = dict((d,ny.ones((comment_Nds[d], J))*(1.0/J)) for d in xrange(D))
    gammaC = ny.ones((D, J)) * (1.0 / J)
    initialize_random(gammaC)

    y = ny.ones((D,)) * 0.5

    eta = ny.ones((K+J,)) * 0.15
    initialize_random(eta)
    print 'eta start: {0}'.format(eta)
    sigma_squared = 1.0

    iterations = 0
    elbo = INITIAL_ELBO
    last_elbo = INITIAL_ELBO - 100

    while elbo_did_not_converge(elbo, last_elbo):
        for d, (document, comment) in enumerate(izip(documents,comments)):
            ### E-step ###
            do_E_step(d, document, comment, alphaD, alphaC, betaD, betaC, gammaD[d], gammaC[d], phiD[d], phiC[d], y, eta, sigma_squared)

        ### M-step: ###
        print 'updating betas..'
        # update betaD for documents first
        recalculate_beta(documents, betaD, phiD)
        print 'comments..'
        # update betaC for comments next
        recalculate_beta(comments, betaC, phiC)

        print 'eta sigma'
        # update response variable gaussian global parameters
        sigma_squared = recalculate_eta_sigma(eta, y, phiD, phiC)

        last_elbo = elbo
        elbo = calculate_global_elbo(documents, comments, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y, eta, sigma_squared)

        print '{1} GLOBAL ELBO: {0}'.format(elbo, iterations)

        # todo: maybe write all these vars every iteration (or every 10) ?

        iterations += 1
    return {'iterations': iterations, 
            'elbo': elbo, 
            'y': y, 
            'eta': eta, 'sigma_squared': sigma_squared, 
            'beta': (betaD, betaC), 
            'gamma': (gammaD, gammaC), 
            'phi': (phiD, phiC), }

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
    big_phi = ny.zeros((n1 + n2, k1 + k2))
    big_phi[0:n1,0:k1] = phi1
    big_phi[n1:n1+n2,k1:k1+k2] = phi2
    return big_phi
    """
    big_phi = dict()
    assert len(phi1) == len(phi2)
    for d in xrange(len(phi1)):
        (n1, k1) = phi1[d].shape
        (n2, k2) = phi2[d].shape
        big_phi[d] = ny.zeros((n1 + n2, k1 + k2))
        big_phi[d][0:n1,0:k1] = phi1[d]
        big_phi[d][n1:n1+n2,k1:k1+k2] = phi2[d]
    return big_phi
    """


def recalculate_eta_sigma(eta, y, phi1, phi2):
    """
        Accepts eta (K+J)-size vector,
            also y (a D-size vector of reals),
            also two phi dictionaries of NxK matrices.
        Returns new sigma squared update (a double).

        ηnew ← (E[ATA])-1 E[A]Ty
        σ2new ← (1/D) {yTy - yTE[A]ηnew}

        (Note that A is the D X (K + J) matrix whose rows are the vectors ZdT for document and comment concatenated.)
        (Also note that the dth row of E[A] is φd, and E[ATA] = Σd E[ZdZdT] .)
        (Also, note that E[Z] = φ := (1/N)Σnφn, and E[ZdZdT] = (1/N2)(ΣnΣm!=nφd,nφd,mT  +  Σndiag{φd,n})
    """
    assert len(phi1) == len(phi2)
    D = len(phi1)

    big_phis = dict()
    for d in xrange(D):
        big_phis[d] = calculate_big_phi(phi1[d], phi2[d])
    (Ndc, KJ) = big_phis[0].shape

    E_A = ny.zeros((D, KJ))
    for d in xrange(D):
        E_A[d] = calculate_EZ(big_phis[d])
  
    E_ATA_inverse = calculate_E_ATA_inverse(big_phis)

    eta[:] = ny.dot(ny.dot(E_ATA_inverse, E_A.T), y)
    new_sigma_squared = (1.0 / D) * (ny.dot(y, y) - (ny.dot(ny.dot(y, E_A), eta)))
    return new_sigma_squared

def calculate_EZZT(big_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[ZdZdT].
        Returns the final matrix ((K+J) x (K+J)).

        (Also, E[ZdZdT] = (1/N2)(ΣNΣm!=nφd,nφd,mT  +  ΣNdiag{φd,n})
    """
    (Ndc, KJ) = big_phi.shape
    inner_sum = ny.zeros((KJ, KJ))
    for n in xrange(Ndc):
        for m in xrange(Ndc):
            if n != m:
                inner_sum += (big_phi[n] * big_phi[m].T)
    for n in xrange(Ndc):
        inner_sum += ny.diag(big_phi[n])
    inner_sum /= (Ndc * Ndc)
    return inner_sum

def calculate_E_ATA_inverse(big_phis):
    """Accepts number of documents, 
        and a big_phi matrix of size (Nd + Nc, K + J).
        Returns a new matrix which is inverse of E([ATA]) of size (K+J,K+J).

        (Note that A is the D X (K + J) matrix whose rows are the vectors ZdT for document and comment concatenated.)
        (Also note that the dth row of E[A] is φd, and E[ATA] = Σd E[ZdZdT] .)
    """
    D = len(big_phis)
    (Ndc, KJ) = big_phis[0].shape
    E_ATA = ny.zeros((KJ, KJ))
    for d in xrange(D):
        E_ATA += calculate_EZZT(big_phis[d])
    return ny.linalg.inv(E_ATA)

def update_gamma_lda_E_step(alpha, phi, gamma):
    """
     Accepts:
        gamma and alpha are K-size vectors.
        Phi is an NxK vector.
     Returns gamma.

     update gamma: γnew ← α + Σnφn
    """
    assert phi.shape[1] == len(gamma)
    gamma[:] = alpha + ny.sum(phi, axis=0)
    return gamma

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
     
     Note that E[log p(wn|β1:K)] = βTwn
    """
    (N, K) = phi.shape
    assert len(eta) == K
    assert len(gamma) == K
    assert beta.shape[0] == K

    phi_sum = ny.sum(phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = dirichlet_expectation(gamma)
    assert len(ElogTheta) == K

    for n,word,count in iterwords(text):
        phi_minus_n = phi_sum - phi[n]
        assert len(phi_minus_n) == K

        pB = ny.log(beta[:,word])
        pC = (1.0 * y_d / Ns * eta)  
        pD = ((-1.0 / (2 * N * Ns)) * 
                (((2 * ny.dot(eta, phi_minus_n) * eta) + (eta * eta)))
                            )
        assert len(pB) == K
        assert len(pC) == K
        assert len(pD) == K
        phi[n,:] = ElogTheta + pB + pC + pD

    ny.exp(phi, phi)
    row_normalize(phi)
    return phi

def do_E_step(d, document, comment, 
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

    # initialize_uniform(phiD)
    # initialize_uniform(phiC)
    initialize_random(phiD)
    initialize_random(phiC)

    local_elbo, local_last_elbo = 0, 0
    print "starting E step"
    i = 0

    local_elbo = INITIAL_ELBO
    last_local_elbo = INITIAL_ELBO - 100
    while elbo_did_not_converge(local_elbo, last_local_elbo):
        # update gammas
        update_gamma_lda_E_step(alphaD, phiD, gammaD)
        update_gamma_lda_E_step(alphaC, phiC, gammaC)

        # update phis (note we have to pass the right part of eta!)
        update_phi_lda_E_step(document, phiD, gammaD, betaD, y[d], eta[:Kd], sigma_squared)
        update_phi_lda_E_step(comment, phiC, gammaC, betaC, y[d], eta[Kd:], sigma_squared)

        # update the response variable
        # y = ηTE[Z] = ηTφ      [  where φ = 1/N * Σnφn   ]
        y[d] = ny.dot(eta, calculate_EZ(calculate_big_phi(phiD, phiC)))

        # calculate new ELBO
        last_local_elbo = local_elbo
        local_elbo = calculate_local_elbo(document, comment, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y[d], eta, sigma_squared)
        i += 1

        print {'beta': (betaD, betaC), 'gamma': (gammaD, gammaC), 'phi': (phiD, phiC), 'y': y, 'eta': eta}
        print "e-step iteration {0} ELBO: {1}".format(i, local_elbo)
    print "done e-step: {0} iterations ELBO: {1}".format(i, local_elbo)

def iterwords(text):
    """Accepts an unchanging dictionary.
        Yields a generator of n,word,count triplets to keep track
        of which word corresponds to the "nth" word in each document.
    """
    n = 0
    for word,count in text.iteritems():
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
    elbo += elbo_lda_terms(alphaD, gammaD, phiD, betaD, document)
    elbo += elbo_lda_terms(alphaC, gammaC, phiC, betaC, comment)

    big_phi = calculate_big_phi(phiD, phiC)
    elbo += elbo_slda_y(y, eta, big_phi, sigma_squared)

    elbo += elbo_entropy(phiD, phiC, gammaD, gammaC)
    return elbo

def elbo_entropy(phiD, phiC, gammaD, gammaC):
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
    elbo += (-1 * ny.sum(phi * ny.log(phi)))

    elbo += -1 * ny.log(GAMMA(ny.sum(gamma)))
    elbo += ny.sum(ny.log(GAMMA(gamma)))

    ElogTheta = dirichlet_expectation(gamma)
    assert ElogTheta.shape == gamma.shape
    elbo += -1 * sum((gamma - 1) * ElogTheta)

    return elbo



# todo: can be made more efficient (look at structure of big phi)
def calculate_EZ(big_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[Zd].
        Returns the final matrix ((K+J) x (K+J)).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc,KJ = big_phi.shape
    return ny.sum(big_phi, axis=0) / Ndc

def elbo_slda_y(y, eta, big_phi, sigma_squared):
    """
    Calculates some terms in the elbo for a document.
    Same as in sLDA.

    E[log p(y|Z1:N,η,σ2)] = (–1/2)log 2πσ2 – (1/2σ2)[y2– 2yηTE[Z] + ηTE[ZZT]η]
    """
    elbo = 0.0
    ss = sigma_squared
    elbo += (-0.5) * ny.log(2 * ny.pi * ss)
    
    ez = calculate_EZ(big_phi)
    ezzt = calculate_EZZT(big_phi)
    nEZZTn = ny.dot(ny.dot(eta, ezzt), eta)
    elbo += (-0.5 * ss) * (y*y - (2 * y * ny.dot(eta, ez)) + nEZZTn)
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
    elbo += ny.log(GAMMA(ny.sum(alpha))) - ny.sum(ny.log(GAMMA(alpha)))

    ElogTheta = dirichlet_expectation(gamma)
    assert len(ElogTheta) == len(alpha)
    assert ElogTheta.shape == alpha.shape
    elbo += ny.sum((alpha - 1) * ElogTheta)

    for n,word,count in iterwords(document):
        for k in xrange(K):
            # E[log p(Zn|θ)] = ΣKφn,kE[log θk]
            elbo += phi[n,k] * ElogTheta[k]
            # E[log p(wn|Zn,β1:K)]  = ΣKφn,klog βk,Wn
            elbo += phi[n,k] * beta[k,word]
    return elbo


def calculate_global_elbo(documents, comments, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y, eta, sigma_squared):
    """Given all of the parametes.
        Calculate the evidence lower bound.
        Helps you know when convergence happens.
    """
    return ny.sum(calculate_local_elbo(documents[d], comments[d], alphaD, alphaC, betaD, betaC, gammaD[d], gammaC[d], phiD[d], phiC[d], y[d], eta, sigma_squared) for d in xrange(len(documents)))

            
if __name__=='__main__':
    # documents are 2-tuples of document, comment
    noisy_test_data = ([
                 {1:1, 2:1, 3:1, 5:1,}, 
                 {0:1, 2:1, 3:1, 4:1,},
                 {1:1, 2:1, 4:1, 5:1,},
                 {5:1, 6:1, 7:1, 9:1,},
                 {5:1, 6:1, 7:1, 9:1,},
                 {5:1, 6:1, 7:1, 8:1,},
                ],[
                 {5:1, 6:1, 8:1, 9:1},
                 {3:1, 5:1, 8:1, 9:1},
                 {0:1, 6:1, 7:1, 8:1},
                 {0:1, 2:1, 3:1, 7:1},
                 {0:1, 1:1, 3:1, 5:1},
                 {0:1, 2:1, 3:1, 4:1},
                ])
    test_data = ([
                 {0:1, 2:2, 3:1, 4:1,},
                 {0:1, 2:1, 3:2, 4:3,},
                 {0:1, 2:3, 3:3, 4:1,},
                 {5:1, 6:2, 8:1, 9:3,},
                 {5:1, 6:2, 8:1, 9:1,},
                 {5:2, 6:1, 8:1, 9:1,},
                 ],
                 [
                 {0:2, 1:1, 3:3, 4:1,},
                 {0:1, 1:2, 3:2, 4:1,},
                 {0:1, 1:1, 3:2, 4:2,},
                 {5:3, 6:1, 7:2, 9:1,},
                 {5:1, 6:2, 7:1, 9:1,},
                 {5:1, 6:1, 7:1, 9:2,},
                ])

    try:
        output = run_em(test_data)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

