#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import izip,chain

try:
    import numpypy as ny
except:
    import numpy as ny

from scipy.special import psi
from scipy.special import gammaln

ny.seterr(invalid='raise')

INITIAL_ELBO = float('-inf')

final_output = {}

def dirichlet_expectation(alpha):
    """
    From Matt Hoffman:
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    #assert len(alpha.shape) == 1 # jperla: not sure what else it does
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
        if ny.abs(elbo - last_elbo) < criterion:
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
    beta[:,:] = ny.zeros(beta.shape)
    for d in xrange(D):
        for n,word,count in iterwords(text[d]):
            for k in xrange(K):
                beta[k,word] += phi[d][n][k]
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

    vocab = max(chain(*[[w[0] for w in d] for d in documents]))
    vocab = max(vocab, max(chain(*[[w[0] for w in c] for c in comments])))
    W = vocab+1
    D = len(documents)
    K = 10
    J = 20

    # give at least more documents than topics
    # so that it's not singular
    assert D > (K+J)

    # calculate number of words per document/comment
    document_Nds = [sum(w[1] for w in d) for d in documents]
    comment_Nds = [sum(w[1] for w in c) for c in comments]

    # "it suffices to fix alpha to uniform 1/K"
    # initialize to ones so that the topics are more evenly distributed
    # good for small datasets
    alphaD = ny.ones((K,)) * (3.0 / K)
    alphaC = ny.ones((J,)) * (3.0 / J)

    # Initialize the variational distribution q(beta|lambda)
    betaD = initialize_beta(K, W)
    betaC = initialize_beta(J, W)

    phiD = [(ny.ones((document_Nds[d], K))*(1.0/K)) for d in xrange(D)]
    phiC = [(ny.ones((comment_Nds[d], J))*(1.0/J)) for d in xrange(D)]

    gammaD = ny.ones((D, K)) * (1.0 / K)
    initialize_random(gammaD)
    gammaC = ny.ones((D, J)) * (1.0 / J)
    initialize_random(gammaC)

    y = ny.random.normal(0.0, 2.0, (D,))
    print 'y start: {0}'.format(y)

    eta = ny.random.normal(0.0, 3.0, (K+J,))
    sigma_squared = 10.0

    # EXPERIMENT
    # Let's initialize eta so that it agrees with y at start
    # hopefully this will keep y closer to gaussian centered at 0
    recalculate_eta_sigma(eta, y, phiD, phiC)

    print 'eta start: {0}'.format(eta)

    iterations = 0
    elbo = INITIAL_ELBO
    last_elbo = INITIAL_ELBO - 100
    local_i = 0
    #for globaliternum in xrange(100):
    while elbo_did_not_converge(elbo, last_elbo, iterations, criterion=0.1):
        for d, (document, comment) in enumerate(izip(documents,comments)):
            ### E-step ###
            local_i = do_E_step(iterations, d, document, comment, alphaD, alphaC, betaD, betaC, gammaD[d], gammaC[d], phiD[d], phiC[d], y, eta, sigma_squared)

        ### M-step: ###
        print 'updating betas..'
        # update betaD for documents first
        recalculate_beta(documents, betaD, phiD)
        print 'comments..'
        # update betaC for comments next
        recalculate_beta(comments, betaC, phiC)

        print 'eta sigma...'
        # update response variable gaussian global parameters
        sigma_squared = recalculate_eta_sigma(eta, y, phiD, phiC)

        print 'will calculate elbo...'
        last_elbo = elbo
        elbo = calculate_global_elbo(documents, comments, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y, eta, sigma_squared)

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
        print final_output
        print 'y: %s' % y
        print 'eta: %s' % eta
        print 'ss: %s' % sigma_squared

        print '{1} ({2} per doc) GLOBAL ELBO: {0}'.format(elbo, iterations, local_i)

    return final_output

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
    E_A = ny.zeros((D, KJ))
    for d in xrange(D):
        E_A[d,:] = calculate_EZ_from_small_phis(phi1[d], phi2[d])
  
    #print 'inverse...'
    E_ATA_inverse = calculate_E_ATA_inverse(phi1, phi2)

    #print 'new eta...'
    new_eta = ny.dot(ny.dot(E_ATA_inverse, E_A.T), y)
    if ny.sum(ny.abs(new_eta)) > (KJ * KJ * 5):
        print 'ETA is GOING CRAZY {0}'.format(eta)
        print 'aborting the update!!!'
    else:
        eta[:] = new_eta
    
    # todo: don't do this later
    # keep sigma squared fix
    #import pdb; pdb.set_trace()
    #new_sigma_squared = (1.0 / D) * (ny.dot(y, y) - ny.dot(ny.dot(ny.dot(ny.dot(y, E_A), E_ATA_inverse), E_A.T), y))
    new_sigma_squared = 1.0
    return new_sigma_squared


# todo: can be made more efficient (look at structure of big phi)
def calculate_EZ_from_small_phis(phi1, phi2):
    """
        Accepts a two small phi matrices (like (NdxK) and (NcxJ))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc = phi1.shape[0] + phi2.shape[0]
    ez = ny.concatenate((ny.sum(phi1, axis=0), ny.sum(phi2, axis=0)), axis=1)
    return ez / Ndc
    
def calculate_EZ(big_phi):
    """
        Accepts a big phi matrix (like ((Nd+Nc) x (K+J))
        Calculates E[Zd].
        Returns the final vector (K+J).

        E[Z] = φ := (1/N)ΣNφn
    """
    Ndc,KJ = big_phi.shape
    return ny.sum(big_phi, axis=0) / Ndc

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
    inner_sum = ny.zeros((KJ, KJ))

    tl = ny.dot(phi1.T, phi1) * (Nd - 1)
    inner_sum[:K,:K] = tl 
    br = ny.dot(phi2.T, phi2) * (Nc - 1)
    inner_sum[K:,K:] = br

    inner_sum[K:,:K] = ny.ones((J,K))
    inner_sum[:K,K:] = ny.ones((K,J))

    big_phi_sum = ny.concatenate((ny.sum(phi1, axis=0),
                                  ny.sum(phi2, axis=0)), axis=1)

    inner_sum += ny.diag(big_phi_sum)

    inner_sum /= (Ndc * Ndc)
    return inner_sum

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
                new_matrix = (ny.matrix(big_phi[n]).T * ny.matrix(big_phi[m]))
                assert new_matrix.shape == inner_sum.shape
                inner_sum += new_matrix
    inner_sum += ny.diag(ny.sum(big_phi, axis=0))

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
    E_ATA = ny.sum(calculate_EZZT_from_small_phis(phi1[d], phi2[d]) for d in xrange(D))
    assert E_ATA.shape == (KJ, KJ)
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

    phi_sum = ny.sum(phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = dirichlet_expectation(gamma)
    assert len(ElogTheta) == K

    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = (eta * eta)
    front = (-1.0 / (2 * N * Ns))

    for n,word,count in iterwords(text):
        phi_sum -= phi[n]
        assert len(phi_sum) == K

        pB = ny.log(beta[:,word])
        pD = (front * (((2 * ny.dot(eta, phi_sum) * eta) + eta_dot_eta))
                            )
        assert len(pB) == K
        assert len(pC) == K
        assert len(pD) == K

        # must exponentiate and sum immediately!
        phi[n,:] = ny.exp(ElogTheta + pB + pC + pD)
        phi[n,:] /= ny.sum(phi[n,:])

        # add this back into the sum
        # unlike in LDA, this cannot be computed in parallel
        phi_sum += phi[n]

    return phi

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
     
     Note that E[log p(wn|β1:K)] = log βTwn
    """
    (N, K) = phi.shape

    phi_sum = ny.sum(phi, axis=0)
    Ns = (N * sigma_squared)
    ElogTheta = dirichlet_expectation(gamma)

    front = (-1.0 / (2 * N * Ns))
    pC = (1.0 * y_d / Ns * eta)  
    eta_dot_eta = front * (eta * eta)
    const = ElogTheta + pC + eta_dot_eta

    right_eta_times_const = (front * 2 * eta)

    for n,word,count in iterwords(text):
        phi_sum -= phi[n]

        pB = ny.log(beta[:,word])
        pD = (ny.dot(eta, phi_sum) * right_eta_times_const) 

        # must exponentiate and normalize immediately!
        phi[n,:] = ny.exp(pB + pD + const)
        phi[n,:] /= ny.sum(phi[n,:])

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
    #print "starting E step on doc {0}".format(d)
    i = 0

    local_elbo = INITIAL_ELBO
    last_local_elbo = INITIAL_ELBO - 100
    while elbo_did_not_converge(local_elbo, last_local_elbo, i, criterion=0.1):
        # print 'will update gamma...'
        # update gammas
        update_gamma_lda_E_step(alphaD, phiD, gammaD)
        update_gamma_lda_E_step(alphaC, phiC, gammaC)

        #print 'will update phis...'
        # update phis (note we have to pass the right part of eta!)
        update_phi_lda_E_step(document, phiD, gammaD, betaD, y[d], eta[:Kd], sigma_squared)
        update_phi_lda_E_step(comment, phiC, gammaC, betaC, y[d], eta[Kd:], sigma_squared)

        #print 'will calculate y...'
        # update the response variable
        # y = ηTE[Z] = ηTφ      [  where φ = 1/N * Σnφn   ]
        y[d] = ny.dot(eta, calculate_EZ_from_small_phis(phiD, phiC))

        #print 'will calculate elbo...'
        # calculate new ELBO
        last_local_elbo = local_elbo
        local_elbo = calculate_local_elbo(document, comment, alphaD, alphaC, betaD, betaC, gammaD, gammaC, phiD, phiC, y[d], eta, sigma_squared)
        '''
        '''
        i += 1

        #print {'beta': (betaD, betaC), 'gamma': (gammaD, gammaC), 'phi': (phiD, phiC), 'y': y, 'eta': eta}
        #print "{2}: e-step iteration {0} ELBO: {1}".format(i, local_elbo, global_iteration)
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
    elbo += -1 * ny.sum(phi * ny.log(phi))

    elbo += -1 * gammaln(ny.sum(gamma))
    elbo += ny.sum(gammaln(gamma))

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
    elbo += (-0.5) * ny.log(2 * ny.pi * ss)
    
    ez = calculate_EZ_from_small_phis(phiD, phiC)
    ezzt = calculate_EZZT_from_small_phis(phiD, phiC)
    nEZZTn = ny.dot(ny.dot(eta, ezzt), eta)
    elbo += (-0.5 / ss) * (y*y - (2 * y * ny.dot(eta, ez)) + nEZZTn)
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
    elbo += gammaln(ny.sum(alpha)) - ny.sum(gammaln(alpha))

    ElogTheta = dirichlet_expectation(gamma)
    #assert len(ElogTheta) == len(alpha)
    #assert ElogTheta.shape == alpha.shape
    elbo += ny.sum((alpha - 1) * ElogTheta)

    for n,word,count in iterwords(document):
        # E[log p(Zn|θ)] = ΣKφn,kE[log θk]
        # E[log p(wn|Zn,β1:K)]  = ΣKφn,klog βk,Wn

        # optimization:
        # E[log p(Zn|θ)] + E[log p(wn|Zn,β1:K)] = ΣKφn,k(E[log θk] + log βk,Wn)
        elbo += ny.sum(phi[n] * (ElogTheta + ny.log(beta[:,word])))
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
    numdocs = 100
    limit_words = 500
    docs = jsondata.read('documents.dc.nyt.json')[:numdocs]
    docs = [d[:limit_words] for d in docs]
    comments = jsondata.read('comments.dc.nyt.json')[:numdocs]
    comments = [c[:limit_words] for c in comments]

    print '%s total words in documents' % sum(len(d) for d in docs)
    print '%s total words in comments' % sum(len(c) for c in comments)

    real_data = [docs, comments]

    try:
        output = run_em(real_data)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

