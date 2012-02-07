"""
    Tests for topiclib library
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
# -*- coding: utf-8 -*-

import numpy as np

import topiclib as lm
import graphlib

from test_graphlib import same

def test_doc_to_array():
    text = [(0,1), (1,1)]
    out = lm.doc_to_array(text)
    answer = np.array([0, 1])
    assert same(out, answer)

    doc1 = [(1,3), (2,2), (0,1)]
    out = lm.doc_to_array(doc1)
    answer = np.array([1, 1, 1, 2, 2, 0])
    assert same(out, answer)

def test_iterwords():
    doc0 = [(0,3), (1,1)]
    doc1 = [(1,3), (2,2),(0,1)]
   
    out = list(lm.iterwords(doc0))
    answer = [
              (0, 0, 3),
              (1, 0, 3),
              (2, 0, 3),
              (3, 1, 1),
             ]
    assert out == answer

    out2 = list(lm.iterwords(doc1))
    answer2 = [
              (0, 1, 3),
              (1, 1, 3),
              (2, 1, 3),
              (3, 2, 2),
              (4, 2, 2),
              (5, 0, 1),
             ]
    assert out2 == answer2

def test_initialize_beta():
    out = lm.initialize_beta(3, 4)
    assert out.shape == (3,4)

    sumrows = np.sum(out, axis=1)
    assert same(sumrows, np.ones(out.shape[0]))

    # test the log version
    out = lm.initialize_log_beta(3, 4)
    assert out.shape == (3,4)

    sumrows = lm.logsumexp(out, axis=1)
    assert same(np.exp(sumrows), np.ones(out.shape[0]))



def test_lda_recalculate_beta():
    K = 2
    W = 3

    doc0 = [(0,3), (1,1)]
    doc1 = [(1,3), (2,2), (0,1)]
    text = [doc0,doc1]

    text = [doc0,doc1]
    beta = np.empty((K,W), dtype=float)
    out = beta.copy()

    phi0 = np.zeros((sum([d[1] for d in doc0]), K))
    # two to topic one (word 0)
    # two to topic two (word 1)
    phi0[0][0] = 1
    phi0[1][0] = 1
    phi0[2][1] = 1
    phi0[3][1] = 1

    phi1 = np.zeros((sum([d[1] for d in doc1]), K))
    phi1[0][1] = 1
    phi1[1][1] = 1
    phi1[2][1] = 1
    phi1[3][0] = 1
    phi1[4][1] = 1
    phi1[5][0] = 1

    phi = [phi0, phi1]

    answer = np.array([[0.75, 0.0, 0.25],
                       [1.0/6, 2.0/3, 1.0/6]])

    assert out.shape == (2,3)
    lm.lda_recalculate_beta(text, out, phi)
    assert out.shape == (2,3)

    assert not same(beta, out)
    assert same(out, answer)

    # now test on docarray
    out = beta.copy()
    assert out.shape == (2,3)
    lm.lda_recalculate_beta([lm.doc_to_array(t) for t in text], out, phi)
    assert out.shape == (2,3)

    assert not same(beta, out)
    assert same(out, answer)


    # test log space
    log_out = np.log(out)
    log_phi = [np.log(p) for p in phi]
    assert log_out.shape == (2,3)
    lm.lda_recalculate_log_beta(text, log_out, log_phi)
    assert log_out.shape == (2,3)

    assert not same(beta, np.exp(log_out))
    assert same(np.exp(log_out), answer)

def test_calculate_big_phi():
    a = np.ones((2,3))
    b = np.ones((6,4))

    a[0, 2] = 5
    b[4,1] = 8

    out = lm.calculate_big_phi(a, b)
    answer = np.array([
                       [1, 1, 5, 0, 0, 0, 0,],
                       [1, 1, 1, 0, 0, 0, 0,],
                       [0, 0, 0, 1, 1, 1, 1,],
                       [0, 0, 0, 1, 1, 1, 1,],
                       [0, 0, 0, 1, 1, 1, 1,],
                       [0, 0, 0, 1, 1, 1, 1,],
                       [0, 0, 0, 1, 8, 1, 1,],
                       [0, 0, 0, 1, 1, 1, 1,],
                      ])
    assert same(out, answer)

phi1 = [graphlib.row_normalize(np.ones((2,3))), ]
phi2 = [graphlib.row_normalize(np.ones((3,2))), ]
log_phi1, log_phi2 = np.log(phi1[0]), np.log(phi2[0])

def test_calculate_EZ():
    big_phi = lm.calculate_big_phi(phi1[0], phi2[0])
    out = lm.calculate_EZ(big_phi)
    answer = (1.0 / 25) * np.array([30.0/9, 30.0/9, 30.0/9, 7.5, 7.5])
    assert same(out, answer)

    out = lm.calculate_EZ_from_small_phis(phi1[0], phi2[0])
    assert same(out, answer)

    # now test log phis
    big_log_phi = lm.calculate_big_log_phi(log_phi1, log_phi2)
    out = lm.calculate_EZ_from_big_log_phi(big_log_phi)
    assert same(out, np.log(answer))

    out = lm.calculate_EZ_from_small_log_phis(log_phi1, log_phi2)
    assert same(out, np.log(answer))

def test_calculate_EZZT():
    big_phi = lm.calculate_big_phi(phi1[0], phi2[0])
    out = lm.calculate_EZZT(big_phi)

    e = 8.0 / 9.0
    t = 2.0 / 9.0
    h = 3.0 / 2.0
    answer = (1.0 / 25) * np.array([
                [e, t, t, 1, 1],
                [t, e, t, 1, 1],
                [t, t, e, 1, 1],
                [1, 1, 1, 3, h],
                [1, 1, 1, h, 3],
               ])
    assert same(out, answer)

    out = lm.calculate_EZZT_from_small_phis(phi1[0], phi2[0])
    assert same(out, answer)

    # try it on logs
    out = lm.calculate_EZZT_from_small_log_phis(log_phi1, log_phi2)
    assert same(out, np.log(answer))

    # now try a harder random matrix
    r1 = answer.copy()
    r1[0,0] = 5
    r1[1,1] = 9
    r1 = graphlib.row_normalize(r1)
    r2 = r1.copy()
    r1[1,1] = 2
    r1[1,0] = 1
    r1 = graphlib.row_normalize(r1)

    big_phi = lm.calculate_big_phi(r1, r2)
    answer = lm.calculate_EZZT(big_phi)
    out = lm.calculate_EZZT_from_small_phis(r1, r2)
    assert same(out, answer)

    # test out same anwer on logs
    out = lm.calculate_EZZT_from_small_log_phis(np.log(r1), np.log(r2))
    assert same(out, np.log(answer))

def test_lda_update_gamma():
    K = 3
    phi = np.array([
                    [0.75, 0.25,   0],
                    [ 0.5,    0, 0.5],
                    [ 0.3,  0.3, 0.4],
                   ])
    alpha = np.array([0.3, 2.3, 0.8])
    gamma = np.zeros((K,))
    out = gamma.copy()
    lm.lda_update_gamma(alpha, phi, out)
    answer = alpha + np.array([1.55, 0.55, 0.9])

    assert not same(out, gamma)
    assert same (out, answer)

    out = np.log(gamma.copy())
    lm.lda_update_log_gamma(np.log(alpha), np.log(phi), out)
    assert not same(np.exp(out), gamma)
    assert same (np.exp(out), answer)

def test_slda_update_phi():
    gamma = np.array([3,4,5])
    text = [(0,1), (1,1)]
    beta = np.array([
                     [0.75, 0.25],
                     [0.40, 0.60],
                     [0.10, 0.90],
                    ])
    y_d = -0.5
    eta = np.array([-2.5, 1.6, 0.1])
    sigma_squared = 0.8
    phi = np.array([
                    [0.65, 0.25, 0.10],
                    [0.09, 0.78, 0.13],
                   ])

    """
    update phid:
    φd,n ∝ exp{ E[log θ|γ] + 
                E[log p(wn|β1:K)] + 
                (y / Nσ2) η  — 
                [2(ηTφd,-n)η + (η∘η)] / (2N2σ2) }
    Note that E[log p(wn|β1:K)] = log βTwn
    """
    eta_dot_eta = np.array([6.25, 2.56, 0.01])
    term1 = np.array([-1.51987734, -1.18654401154, -0.93654401154401])
    term2 = np.log(np.array([0.75, 0.40, 0.10]))
    term3 = np.array([0.78125, -0.5, -0.03125])
    term4 = -0.15625 * ((2 * (np.dot(eta, phi[1])) * eta) + eta_dot_eta)

    first_row = np.exp(term1 + term2 + term3 + term4)
    first_row /= np.sum(first_row) # normalize it, then set

    # note that this happens in sequential order, so must use first row, not old phi[0]
    term2 = np.log(np.array([0.25, 0.60, 0.90]))
    term4 = -0.15625 * ((2 * (np.dot(eta, first_row)) * eta) + eta_dot_eta)

    second_row = np.exp(term1 + term2 + term3 + term4)
    answer = np.array([first_row, second_row])

    graphlib.row_normalize(answer)

    out = phi.copy()
    lm.slda_update_phi(text, out, gamma, beta, y_d, eta, sigma_squared)
    assert same(out, answer)

    # test the fast updates; which will be slightly different
    fast_answer = answer.copy()
    fast_answer[1,:] = np.array([0.03422278, 0.26873478, 0.69704244])

    out = phi.copy()
    docarray = lm.doc_to_array([(0,1), (1,1)])
    lm.slda_update_phi(docarray, out, gamma, beta, y_d, eta, sigma_squared)
    
    assert same(out, fast_answer)

def test_elbo():
    # todo: calculate elbo terms
    # do one calculation
    #lm.lda_elbo_terms(document, alphaD, betaD, gammaD, phiD)
    #lm.slda_elbo_y(y, eta, big_phi, sigma_squared)
    #lm.lda_elbo_entropy(phiD, gammaD)
    pass

"""
# todo: need to make a lot more phi matrices
def test_calculate_E_ATA_inverse():
    pass

def test_recalculate_eta_sigma():
    D = 10

    eta = [3.0, 2.0, 1.0, 4.0]
    y = [2.0, 2.0, -2.0, -2.0]
    answer_eta = 0
    answer_sigma_squared = 0
"""
