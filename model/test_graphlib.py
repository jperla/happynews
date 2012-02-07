"""
    Tests for graphlib library
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import numpy as np

import graphlib

def same(a, b):
    print a
    print b
    assert a.shape == b.shape
    return np.allclose(a, b)
    #return np.all(np.abs(a-b) < 0.00000001)

def test_logdot():
    from test_topiclib import phi1,phi2
    log_phi1, log_phi2 = np.log(phi1[0]), np.log(phi2[0])
    assert same(np.dot(phi1[0], phi2[0]), np.exp(graphlib.logdotexp(log_phi1, log_phi2)))

def test_dirichlet_expectation():
    alpha = np.array([3,4,5])
    out = graphlib.dirichlet_expectation(alpha)
    answer = np.array([-1.51987734, -1.18654401154, -0.93654401154401])
    assert same(out, answer)

def test_row_normalize():
    t = np.ones((3,4))
    out = graphlib.row_normalize(t)
    answer = np.ones(t.shape) * 0.25
    assert same(out, answer)

    a = np.array([[ 0.75      ,  0.        ,  0.25      ],
                  [ 0.16666667,  0.66666667,  0.16666667]])
    graphlib.row_normalize(a)
    assert a.shape == (2, 3)


def test_log_row_normalize():
    m = np.log(np.array([[2,2,4], [3,2,1]]))
    answer = np.log(np.array([[0.25, 0.25, 0.5], [0.5, 0.333333333333333, 0.166666666666667]]))

    assert abs(graphlib.logsumexp(m[0,:]) - np.log(8)) < .0000000001
    assert abs(graphlib.logsumexp(m[1,:]) - np.log(6)) < .0000000001
    assert abs(graphlib.logsumexp(answer[0,:])) < .0000000001

    out = graphlib.log_row_normalize(m)
    assert same(out, answer)


def test_initialize_uniform():
    m = np.ones((4,5))
    out = graphlib.initialize_uniform(m)
    answer = np.ones((4,5)) * 0.20
    assert same(out, answer)

    m = np.ones((4,5))
    out = graphlib.initialize_log_uniform(m)
    answer = np.log(np.ones((4,5)) * 0.20)
    assert same(out, answer)

def test_initialize_random():
    original = np.ones((4,7))
    out = original.copy()
    graphlib.initialize_random(out)
    assert original.shape == out.shape

    assert not same(out, original)

    sumrows = np.sum(out, axis=1)
    assert same(sumrows, np.ones(out.shape[0]))

    # now test log of the same
    original = np.ones((4,7))
    out = original.copy()
    graphlib.initialize_log_random(out)
    assert original.shape == out.shape

    assert not same(out, original)

    sumrows = graphlib.logsumexp(out, axis=1)
    assert same(np.exp(sumrows), np.ones(out.shape[0]))


def test_elbo_did_not_converge():
    # non-convergence
    ib = graphlib.INITIAL_ELBO
    assert graphlib.elbo_did_not_converge(-10.0, -20.0, 1, 0.00001)
    assert graphlib.elbo_did_not_converge(-10.0, -11.0, 1, 0.00001)
    assert graphlib.elbo_did_not_converge(-10.0, ib, 1, 0.00001)
    assert graphlib.elbo_did_not_converge(ib, ib, 1, 0.00001)
    assert graphlib.elbo_did_not_converge(ib, 0, 1, 0.00001)

    assert not graphlib.elbo_did_not_converge(-10.0, -11.0, 100, 0.00001)
    assert graphlib.elbo_did_not_converge(-10.0, -11.0, 99, 0.00001, max_iter=100)
    assert not graphlib.elbo_did_not_converge(-10.0, -11.0, 99, 0.00001, max_iter=99)

    assert not graphlib.elbo_did_not_converge(-10.0, -10.0, 1, 0.00001)
    assert not graphlib.elbo_did_not_converge(-10.0000000001, -10.0, 1, 0.00001)

