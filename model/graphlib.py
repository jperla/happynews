#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
from itertools import izip

try:
    import numpy as np
except ImportError:
    import numpypy as np
else:
    np.seterr(invalid='raise')

try:
    from scipy.special import psi
except ImportError:
    from scipypy import psi

INITIAL_ELBO = float('-inf')

def ensure(boolean, message=''):
    """Wrapper for assert. Let's me turn it on/off globally."""
    assert boolean, message

def logsumexp(a, axis=0):
    """Same as sum, but in log space. Compare to logaddexp."""
    return np.logaddexp.reduce(a, axis)

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


def log_row_normalize(m):
    """Does row-normalize in log space.
        All values in m are log probabilities.
    """
    assert len(m.shape) == 2
    lognorm = logsumexp(m, axis=1)
    lognorm.shape = (lognorm.shape[0], 1)

    m -= lognorm
    return m

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
                          criterion=0.001, min_iter=0, max_iter=20):
    """Accepts two elbo doubles.  
        Also accepts the number of iterations already performed in this loop.
        Also accepts convergence criterion: 
            (elbo - last_elbo) < criterion # True to stop
        Finally, accepts 
        Returns boolean.
        Figures out whether the elbo is sufficiently smaller than
            last_elbo.
    """
    if num_iter < min_iter:
        return True

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


class GraphVars(object):
    """Abstract class useful for holding 
        graphical model hidden and observed variables.
    """
    def __init__(self):
        pass

    def num_words_per(self, docs):
        # calculate number of words per document/comment
        if isinstance(docs[0], np.ndarray):
            return [len(d) for d in docs]
        else:
            return [sum(w[1] for w in d) for d in docs]


def run_variational_em(var, e_step_func, m_step_func, global_elbo_func, print_func=None):
    """
    General Variational Expectation Maximization algorithm.

    var: the GraphVars variables state class. This keeps track of the state of optimization.

    e_step_func: performs the expectation step. 
                 Accepts total number of iterations, and GraphVars variables state class.
                 Returns some information about how long the E-step took.
                 In LDA for example, 
                    it will return the number of local iterations 
                    it took to converge at the document level

    m_step_func: performs the maximization step. 
                 Accepts the GraphVars variables state class.
                 Returns nothing.

    global_elbo_func: returns the Evidence Lower Bound for variational EM
                 Accepts the GraphVars variables state class.
                 Returns a double.
                    
    print_func: useful for debugging. optional. Accepts local state. May print things.
                 Accepts the GraphVars variables state class.
                 Returns nothing.

    Algorithm:
    Initialize all the variables (assumed to be already done in var)
    Repeat until the ELBO converges (global_elbo_func):
        Do the E-step:
            For each document d:
                (e_step_func)
                Initialize local variables randomly.
                Repeat until the local ELBO converges:
                    Update all of the local variables
        Do the M-step (m_step_func):
            Update the global parameters.
        Print any variables for debugging (print_func)
    """
    assert var.is_initialized

    global final_output
    final_output = {}

    iterations = 0
    elbo = INITIAL_ELBO
    last_elbo = INITIAL_ELBO - 100
    local_i = 0

    #for globaliternum in xrange(100):
    while elbo_did_not_converge(elbo, last_elbo, 
                                iterations, criterion=0.1, max_iter=100):
        
        ### E-step ###
        local_i = e_step_func(iterations, var)

        m_step_func(var)

        if local_i < 20 and iterations >= 5 and iterations % 2 == 0:
            print 'will calculate global elbo...'
            last_elbo = elbo
            elbo = global_elbo_func(var)
        else:
            print 'skip global elbo in first few iterations...'

        # todo: maybe write all these vars every iteration (or every 10) ?
        iterations += 1

        final_output.update(var.to_dict())
        final_output.update({'iterations': iterations, 'elbo': elbo,})

        #print final_output
        if print_func is not None:
            print_func(var)

        print '{1} ({2} per doc) GLOBAL ELBO: {0}'.format(elbo, iterations, local_i)

    return final_output


########### PYPY-only functions! ###########
this_is_pypy = ('matrix' in dir(np))

def ispypy():
    """Returns a boolean True if pypy is running the program.
        Does this by checking the matrix module, which is not currently implemented.
    """
    return this_is_pypy

def random_normal(mu, sigma, shape):
    """Define my own random normal, since numpypy does not have np.random.normal ."""
    size = shape[0]
    n = np.array([random.gauss(mu, sigma) for i in xrange(size)])
    return n

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
    NOTE: THIS APPEARS TO BE VERY BROKEN
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
    """Calculates matrix[:,array]
    NOTE: THIS APPEARS TO BE VERY BROKEN
    """
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

